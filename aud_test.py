import pandas as pd 
import json
import torch
import time 
import os 
import argparse 
from attrdict import AttrDict
from transformers import BertConfig, BertTokenizer, BertModel, BertForSequenceClassification
from torchmetrics import R2Score
from torchmetrics.classification import F1Score, MulticlassPrecision, MulticlassRecall, MulticlassSpecificity
from sklearn.metrics import precision_score , recall_score , confusion_matrix, f1_score, classification_report
from src import BertDataset, BertProcessor, BertRegressor, BertRegTester, BertClsTester

def main(args):
    with open(os.path.join(args.config_path, args.config_file)) as f:
        training_config = AttrDict(json.load(f))
    
    # set training config  
    training_config.default_path = os.getcwd()
    training_config.data_path = os.path.join(training_config.default_path, args.data_path)
    training_config.log_path = os.path.join(training_config.default_path, args.log_path)
    training_config.base_model = os.path.join(training_config.default_path, args.base_model)
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
    bws_config.max_position_embeddings = 128
    bws_model = BertModel.from_pretrained(os.path.join(training_config.base_model, 'bert-small'), config=bws_config)
    dsm_tokenizer = BertTokenizer.from_pretrained(os.path.join(training_config.base_model, 'bert-mini'), model_max_length=128)
    dsm_config = BertConfig.from_pretrained(os.path.join(training_config.base_model, 'bert-mini', 'bert_config.json'), num_labels=10, output_hidden_states=True)
    dsm_config.max_position_embeddings = 128
    dsm_model = BertForSequenceClassification.from_pretrained(os.path.join(training_config.base_model, 'bert-mini'), config=dsm_tokenizer)
    
    # bws, dsm-5 data process 
    bws_test = pd.read_csv(os.path.join(training_config.data_path, 'bws_score_test.csv'))
    bws_test_file = BertDataset(bws_test)
    bws_processor = BertProcessor(training_config, bws_tokenizer)
    bws_test_dataset = bws_processor.convert_data(bws_test_file)
    bws_test_sampler = bws_processor.shuffle_data(bws_test_dataset, 'test')
    bws_test_dataloader = bws_processor.load_data(bws_test_dataset, bws_test_sampler)

    dsm_test = pd.read_csv(os.path.join(training_config.data_path, 'dsm_samp_test.csv'))
    dsm_test_file = BertDataset(dsm_test)
    dsm_processor = BertProcessor(training_config, dsm_tokenizer)
    dsm_test_dataset = dsm_processor.convert_data(dsm_test_file)
    dsm_test_sampler = dsm_processor.shuffle_data(dsm_test_dataset, 'test')
    dsm_test_dataloader = dsm_processor.load_data(dsm_test_dataset, dsm_test_sampler)

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

    start = time.time()
    bws_pred, bws_true = bws_tester.get_label(bws_test_dataloader, 0)
    print(f'bws predict time: {round(time.time - start, 3)}')
    start2 = time.time()
    dsm_pred, dsm_true = dsm_tester.get_label(dsm_test_dataloader, 0)
    print(f'dsm predict time: {round(time.time - start2, 3)}')

    def RMSELoss(yhat,y):
        return torch.sqrt(torch.mean((yhat-y)**2))
    
    criterion = RMSELoss
    r2score = R2Score 
    bws_rmse = criterion(torch.Tensor(bws_pred), torch.Tensor(bws_true))
    bws_r2 = r2score(torch.Tensor(bws_pred), torch.Tensor(bws_true))
    print(f'bws rmse: {bws_rmse}, bws r2 score: {bws_r2}')

    f1 = F1Score(task="multiclass", num_classes=len(dsm_test.label.unique()))
    precision = MulticlassPrecision(num_classes=len(dsm_test.label.unique()))
    recall = MulticlassRecall(num_classes=len(dsm_test.label.unique()))
    specificity = MulticlassSpecificity(num_classes=len(dsm_test.label.unique()))

    print(f'dsm precision: {precision(torch.Tensor(dsm_pred), torch.Tensor(dsm_true))}')
    print(f'dsm recall: {recall(torch.Tensor(dsm_pred), torch.Tensor(dsm_true))}')
    print(f'dsm specificity: {specificity(torch.Tensor(dsm_pred), torch.Tensor(dsm_true))}')
    print(f'dsm f1-score: {f1(torch.Tensor(dsm_pred), torch.Tensor(dsm_true))}')

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--data_path", type=str, default='data', help='path that have train, val data')
    cli_parser.add_argument("--base_model", type=str, default='base-model', help='path that have pretrained model')
    cli_parser.add_argument("--model_path", type=str, default='models', help='path to save finetuned model')
    cli_parser.add_argument("--config_path", type=str, default='config', help='path that have training config')
    cli_parser.add_argument("--log_path", type=str, default='log', help='path to save training log')
    cli_parser.add_argument("--config_file", type=str, default='training_config.json')
    
    cli_argse = cli_parser.parse_args()
    main(cli_argse)