# save as: generate_ddi_pkl.py

import pandas as pd
import pickle
import argparse

def read_and_save_multiple_datasets(train_csv, val_csv, test_csv, output_file, ddi_id_file):
    datasets = {}
    for split, csv_file in zip(['train', 'val', 'test'], [train_csv, val_csv, test_csv]):
        print(f"Processing {split} dataset: {csv_file}")
        df = pd.read_csv(csv_file)

        id1 = df['id1'].tolist()
        id2 = df['id2'].tolist()
        ddi = df['ddi'].tolist()

        ddi_df = pd.read_csv(ddi_id_file, sep='\t')
        ddi_to_id = {row['ddi']: row['edge_index'] for _, row in ddi_df.iterrows()}

        ddi_id = [ddi_to_id[ddi_value] for ddi_value in ddi]

        datasets[split] = {
            'id1': id1,
            'id2': id2,
            'ddi': ddi,
            'ddi_id': ddi_id
        }

    with open(output_file, 'wb') as outfile:
        pickle.dump(datasets, outfile)

    print(f"Saved processed dataset to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save DDI datasets into a .pkl file.")
    parser.add_argument('--train', required=True, help='Path to train.csv')
    parser.add_argument('--val', required=True, help='Path to val.csv')
    parser.add_argument('--test', required=True, help='Path to test.csv')
    parser.add_argument('--ddi', required=True, help='Path to DDI ID mapping file (TSV)')
    parser.add_argument('--output', required=True, help='Output .pkl file path')

    args = parser.parse_args()

    read_and_save_multiple_datasets(
        train_csv=args.train,
        val_csv=args.val,
        test_csv=args.test,
        output_file=args.output,
        ddi_id_file=args.ddi
    )
