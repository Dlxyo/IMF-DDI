#!/usr/bin/env python3
import argparse
import pandas as pd
import pickle

def read_float_array(file_path):
    '''
    Read a text file where each line contains space-separated floats and return a list of float lists.
    Args:
        file_path (str): Path to the text file.
    Returns:
        List[List[float]]: Parsed float arrays from each line.
    '''
    with open(file_path, 'r') as f:
        return [list(map(float, line.split())) for line in f.readlines()]


def read_and_save_data_4entity(txt_file3, csv_file, output_file):
    '''
    Read node representations for four entity types and save to a pickle file.
    Args:
        txt_file3 (str): Path to the text file containing part3 embeddings.
        csv_file (str): Path to the CSV file with index, enzyme, target, gene, and disease columns.
        output_file (str): Path for the output pickle file.
    '''
    data3 = read_float_array(txt_file3)
    df = pd.read_csv(csv_file)

    ids = df['index'].tolist()
    enzymes = df['enzyme'].tolist()
    targets = df['target'].tolist()
    genes = df['gene'].tolist()
    diseases = df['disease'].tolist()

    node_representations = {
        'id': ids,
        'part3': data3,
        'enzyme': enzymes,
        'target': targets,
        'gene': genes,
        'disease': diseases
    }

    with open(output_file, 'wb') as outfile:
        pickle.dump(node_representations, outfile)

    print(f'4-entity data saved to {output_file}')


def read_and_save_data_entity(txt_file3, csv_file, output_file):
    '''
    Read node representations for two entity types and save to a pickle file.
    Args:
        txt_file3 (str): Path to the text file containing part3 embeddings.
        csv_file (str): Path to the CSV file with index, enzyme, and target columns.
        output_file (str): Path for the output pickle file.
    '''
    data3 = read_float_array(txt_file3)
    df = pd.read_csv(csv_file)

    ids = df['index'].tolist()
    enzymes = df['enzyme'].tolist()
    targets = df['target'].tolist()

    node_representations = {
        'id': ids,
        'part3': data3,
        'enzyme': enzymes,
        'target': targets
    }

    with open(output_file, 'wb') as outfile:
        pickle.dump(node_representations, outfile)

    print(f'Entity data saved to {output_file}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert text embeddings and CSV metadata to pickle with node representations.'
    )
    parser.add_argument('--txt_file3', required=True,
                        help='Path to the text file containing part3 embeddings.')
    parser.add_argument('--csv_file', required=True,
                        help='Path to the CSV file with metadata columns.')
    parser.add_argument('--output_file', required=True,
                        help='Path for the output pickle file.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--four_entity', action='store_true',
                       help='Process index, enzyme, target, gene, and disease columns.')
    group.add_argument('--entity', action='store_true',
                       help='Process index, enzyme, and target columns only.')
    args = parser.parse_args()

    if args.four_entity:
        read_and_save_data_4entity(
            txt_file3=args.txt_file3,
            csv_file=args.csv_file,
            output_file=args.output_file
        )
    elif args.entity:
        read_and_save_data_entity(
            txt_file3=args.txt_file3,
            csv_file=args.csv_file,
            output_file=args.output_file
        )

if __name__ == '__main__':
    main()
