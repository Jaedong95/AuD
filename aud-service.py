import pandas as pd 
import json
import torch
from datetime import datetime
import os 
import argparse 
from attrdict import AttrDict
from transformers import BertConfig, BertTokenizer, BertModel, BertForSequenceClassification
from src import BertDataset, BertProcessor, BertRegressor, BertRegTester, BertClsTester
from src import AuDDB

def main(args):
    global training_config 
    with open(os.path.join(args.config_path, args.db_config)) as f:
        db_config = AttrDict(json.load(f))

    aud_db = AuDDB(db_config)
    aud_db.connect()
    
    with open(os.path.join(args.config_path, args.config_file)) as f:
        training_config = AttrDict(json.load(f))
    
    # set training config  
    training_config.default_path = os.getcwd()
    training_config.base_model = os.path.join(training_config.default_path, args.base_model)
    training_config.data_path = os.path.join(training_config.default_path, args.data_path)
    training_config.model_path = os.path.join(training_config.default_path, args.model_path)
    training_config.config_path = os.path.join(training_config.default_path, args.config_path)
    training_config.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    training_config.pad = 'max_length'

    label = dict()
    label[0] = 'depressed'
    label[1] = 'lethargic'
    label[2] = 'appetite/weight problem'
    label[3] = 'sleep disorder'
    label[4] = 'emotional instability'
    label[5] = 'fatigue'
    label[6] = 'excessive guilt/worthlessness'
    label[7] = 'cognitive problems'
    label[8] = 'suicidal thoughts'
    label[9] = 'daily'

    # load pretrained tokenizer, config, model  
    bws_tokenizer = BertTokenizer.from_pretrained(os.path.join(training_config.base_model, 'bert-small'), model_max_length=128)
    bws_config = BertConfig.from_pretrained(os.path.join(training_config.base_model, 'bert-small', 'bert_config.json'), output_hidden_states=True, output_attentions=True)
    bws_model = BertModel.from_pretrained(os.path.join(training_config.base_model, 'bert-small'), config=bws_config)
    dsm_tokenizer = BertTokenizer.from_pretrained(os.path.join(training_config.base_model, 'bert-mini'), model_max_length=128)
    dsm_config = BertConfig.from_pretrained(os.path.join(training_config.base_model, 'bert-mini', 'bert_config.json'), num_labels=10, output_hidden_states=True, output_attentions=True)
    dsm_model = BertForSequenceClassification.from_pretrained(os.path.join(training_config.base_model, 'bert-mini'), config=dsm_config)
    bws_config.max_position_embeddings = 128
    dsm_config.max_position_embeddings = 128

    bws_processor = BertProcessor(training_config, bws_tokenizer)
    dsm_processor = BertProcessor(training_config, dsm_tokenizer)

    # load fine-tuned model
    bws_reg = BertRegressor(bws_config, bws_model)
    bws_model_name = os.path.join(training_config.model_path, 'BWS.pt')
    bws_reg.load_state_dict(torch.load(bws_model_name, map_location=torch.device('cpu')))
    bws_reg.to(training_config.device)
    bws_tester = BertRegTester(training_config, bws_reg)

    dsm_model_name = os.path.join(training_config.model_path, 'DSM-5.pt')
    dsm_model.load_state_dict(torch.load(dsm_model_name, map_location=torch.device('cpu')))
    dsm_model.to(training_config.device)
    dsm_tester = BertClsTester(training_config, dsm_model)

    flag = 0 
    user = input('who are you ? ')   # save user name 
    
    while flag == 0: 
        if args.service_type == 0: 
            print(f'Analyzing single sentence ... ')
            print(f'If you want to exit, enter 1')
            input_text = input(f'Please input text: ')
            try: 
                assert int(input_text) == 1
                flag = int(input_text)
            except: 
                bws_data = bws_processor.convert_sentence(input_text)
                bws_sampler = bws_processor.shuffle_data(bws_data, 'test')
                bws_loader = bws_processor.load_data(bws_data, bws_sampler)
                bws_pred, _ = bws_tester.get_label(bws_loader, 1)
                bws_score = round(bws_pred[0], 3)
                
                dsm_data = dsm_processor.convert_sentence(input_text)
                dsm_sampler = dsm_processor.shuffle_data(dsm_data, 'test')
                dsm_loader = dsm_processor.load_data(dsm_data, dsm_sampler)
                dsm_pred, _ = dsm_tester.get_label(dsm_loader, 1)
                dsm_label = label[dsm_pred[0]]
                
                tokens = dsm_tester.get_att_toks(input_text, dsm_tokenizer, dsm_model)
                tokens = ', '.join(tokens)
                now = datetime.now()
                aud_db.save_aud_log(user, input_text, bws_score, dsm_label, tokens, now)
        if args.service_type == 1: 
            print('Analayzing data file... file name: user_input.csv')
            data = pd.read_csv(os.path.join(training_config.data_path, 'user_input.csv'))
            data['user'] = user
            conv_file = BertDataset(data)
            bws_data = bws_processor.convert_data(conv_file)
            bws_sampler = bws_processor.shuffle_data(bws_data, 'test')
            bws_loader = bws_processor.load_data(bws_data, bws_sampler)
            bws_pred, _ = bws_tester.get_label(bws_loader, 0)
            data['bws_score'] = [pred for pred in bws_pred]
            data['bws_score'] = data['bws_score'].apply(lambda x: round(x, 3))
            
            dsm_data = dsm_processor.convert_data(conv_file)
            dsm_sampler = dsm_processor.shuffle_data(dsm_data, 'test')
            dsm_loader = dsm_processor.load_data(dsm_data, dsm_sampler)
            dsm_pred, _ = dsm_tester.get_label(dsm_loader, 0)
            dsm_label = label[dsm_pred[0]]
            data['dsm_label'] = [label[pred] for pred in dsm_pred]
            
            token_list = [dsm_tester.get_att_toks(text, dsm_tokenizer, dsm_model) for text in data['text'].values]
            data['tokens'] = token_list 
            data['tokens'] = data['tokens'].apply(lambda x: ', '.join(x).replace(' ', ''))
            data = data[['user', 'text', 'bws_score', 'dsm_label', 'tokens']]
            data.to_csv(os.path.join(training_config.data_path, 'user_input_result.csv'), index=False)
            flag = 1


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--base_model", type=str, default='base-model', help='path that have pretrained model')
    cli_parser.add_argument("--model_path", type=str, default='models', help='path to save finetuned model')
    cli_parser.add_argument("--config_path", type=str, default='config', help='path that have training config')
    cli_parser.add_argument("--config_file", type=str, default='training_config.json')
    cli_parser.add_argument("--data_path", type=str, default='data', help='path that have user_input.csv file')
    cli_parser.add_argument("--db_config", type=str, default='db_config.json')
    cli_parser.add_argument("--service_type", type=bool, default=0, help='analyze user input: 0, data file: 1')
    
    cli_argse = cli_parser.parse_args()
    main(cli_argse)