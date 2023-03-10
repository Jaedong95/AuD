import pandas as pd 
import argparse 
import os 

def main(args):
    default_path = os.getcwd()
    data_path = os.path.join(default_path, args.data_path)
    processed_data = os.path.join(args.data_path, 'processed')
    


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--data_path", type=str, default='data')
    
    cli_argse = cli_parser.parse_args()
    main(cli_argse)