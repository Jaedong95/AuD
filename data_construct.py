import pandas as pd 
import argparse 
import os 
from src import AuDWord2Vec, DSMProcessor

def main(args):
    default_path = os.getcwd()
    data_path = os.path.join(default_path, args.data_path)
    model_path = os.path.join(default_path, args.model_path)
    processed_data = os.path.join(args.data_path, 'processed')
    reddit_year = list(range(2010, 2019))
    aud_w2v = AuDWord2Vec(model_path)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--data_path", type=str, default='data')
    cli_parser.add_argument("--model_path", type=str, default='models')
    cli_argse = cli_parser.parse_args()
    main(cli_argse)