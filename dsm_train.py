import pandas as pd 
import json
import torch
import os 
import argparse 
from attrdict import AttrDict
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from src import BertDataset, BertProcessor, BertClsTrainer

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
    training_config.num_epochs = 500

     # load pretrained tokenizer, config, model  
    tokenizer = BertTokenizer.from_pretrained(os.path.join(training_config.base_model, 'bert-tiny'), model_max_length=128)
    config = BertConfig.from_pretrained(os.path.join(training_config.base_model, 'bert-tiny', 'bert_config.json'), num_labels=10)
    model = BertForSequenceClassification.from_pretrained(os.path.join(training_config.base_model, 'bert-tiny'), config=config)
    config.max_position_embeddings = 128

    # process data 
    X_train = pd.read_csv(os.path.join(training_config.data_path, 'dsm_samp_train.csv'))
    X_val = pd.read_csv(os.path.join(training_config.data_path, 'dsm_samp_val.csv'))
    train_file = BertDataset(X_train)
    val_file = BertDataset(X_val)
    bert_processor = BertProcessor(training_config, tokenizer)
    train_dataset = bert_processor.convert_data(train_file)
    val_dataset = bert_processor.convert_data(val_file)
    train_sampler = bert_processor.shuffle_data(train_dataset, 'train')
    val_sampler = bert_processor.shuffle_data(val_dataset, 'eval')
    train_dataloader = bert_processor.load_data(train_dataset, train_sampler)
    val_dataloader = bert_processor.load_data(val_dataset, val_sampler)

    # train model 
    bert_trainer = BertClsTrainer(config, training_config, model, train_dataloader, val_dataloader)
    bert_trainer.set_seed()
    bert_trainer.train()

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