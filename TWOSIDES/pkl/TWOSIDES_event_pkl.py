#!/usr/bin/env python3
import argparse
import pandas as pd
import pickle
import ast
import torch
from tqdm import tqdm

def process_ddi_entries(ddi_raw, max_ddi=200):
    '''
    Process DDI entries and convert them to PyTorch tensors.
    Args:
        ddi_raw (List[str]): Raw DDI data entries.
        max_ddi (int): Maximum number of DDI types (default: 200).
    Returns:
        List[torch.Tensor]: List of processed DDI tensors.
    '''
    ddi_tensors = []
    for idx, ddi_entry in tqdm(enumerate(ddi_raw), total=len(ddi_raw), desc='Processing DDI entries'):
        # Handle missing or empty DDI entries
        if pd.isna(ddi_entry):
            print('Empty DDI entry found; inserting zero tensor.')
            ddi_tensor = torch.zeros(max_ddi, dtype=torch.float32)
            ddi_tensors.append(ddi_tensor)
            continue
        try:
            ddi_list = ast.literal_eval(ddi_entry)
            if not isinstance(ddi_list, list):
                raise ValueError('Parsed result is not a list.')
            # Convert to tensor
            ddi_tensor = torch.tensor(ddi_list, dtype=torch.float32)
            if ddi_tensor.size(0) != max_ddi:
                print(f'Warning: label length is not {max_ddi}; truncating or padding.')
                new_tensor = torch.zeros(max_ddi, dtype=torch.float32)
                length = min(len(ddi_list), max_ddi)
                new_tensor[:length] = ddi_tensor[:length]
                ddi_tensor = new_tensor
            ddi_tensors.append(ddi_tensor)
            # Print first entry details
            if idx == 0:
                print(f'First DDI entry (raw): {ddi_entry}')
                print(f'First DDI entry (parsed): {ddi_list}')
                print(f'First DDI entry (tensor): {ddi_tensor}')
        except Exception as e:
            print(f'Warning: could not parse DDI "{ddi_entry}"; using zero tensor. Error: {e}')
            ddi_tensors.append(torch.zeros(max_ddi, dtype=torch.float32))
    return ddi_tensors

def read_and_save_multiple_datasets(train_csv, val_csv, test_csv, output_file, max_ddi=200):
    '''
    Read train, validation, and test CSV files, process DDI entries, and save to a pickle file.
    Args:
        train_csv (str): Path to the training CSV file.
        val_csv (str): Path to the validation CSV file.
        test_csv (str): Path to the test CSV file.
        output_file (str): Path to the output pickle file.
        max_ddi (int): Maximum number of DDI types (default: 200).
    '''
    datasets = {}
    for split, csv_file in zip(['train', 'val', 'test'], [train_csv, val_csv, test_csv]):
        print(f'Processing {split} dataset: {csv_file}')
        df = pd.read_csv(csv_file)
        id1 = df['id1'].tolist()
        id2 = df['id2'].tolist()
        ddi_raw = df['ddi'].tolist()
        polarity = df['polarity'].tolist()
        ddi_tensors = process_ddi_entries(ddi_raw, max_ddi)
        datasets[split] = {
            'id1': id1,
            'id2': id2,
            'ddi': ddi_tensors,
            'polarity': polarity
        }
    # Save to pickle
    with open(output_file, 'wb') as outfile:
        pickle.dump(datasets, outfile)
    print(f'Data saved to {output_file}')

def main():
    parser = argparse.ArgumentParser(description='Convert DDI CSV datasets to pickle with PyTorch tensors.')
    parser.add_argument('--train_csv', required=True, help='Path to the training CSV file.')
    parser.add_argument('--val_csv', required=True, help='Path to the validation CSV file.')
    parser.add_argument('--test_csv', required=True, help='Path to the test CSV file.')
    parser.add_argument('--output_file', required=True, help='Path for the output pickle file.')
    parser.add_argument('--max_ddi', type=int, default=200, help='Maximum number of DDI types.')
    args = parser.parse_args()
    read_and_save_multiple_datasets(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        output_file=args.output_file,
        max_ddi=args.max_ddi
    )

if __name__ == '__main__':
    main()