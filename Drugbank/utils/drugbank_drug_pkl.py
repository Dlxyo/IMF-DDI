import pandas as pd
import pickle
import argparse

def read_float_array(file_path):
    with open(file_path, 'r') as f:
        return [list(map(float, line.split())) for line in f.readlines()]

def read_and_save_data(csv_file, txt_file3, output_file, full=False):
    data3 = read_float_array(txt_file3)
    df = pd.read_csv(csv_file)

    node_representations = {
        'id': df['id'].tolist(),
        'part3': data3,
        'enzyme': df['enzyme'].tolist(),
        'target': df['target'].tolist(),
    }

    if full:
        node_representations.update({
            'gene': df['gene'].tolist(),
            'disease': df['disease'].tolist(),
            'pathway': df['pathway'].tolist(),
        })

    with open(output_file, 'wb') as outfile:
        pickle.dump(node_representations, outfile)

    print(f"Saved to: {output_file}")
    print(f"Keys: {list(node_representations.keys())}")
    print(f"Sample IDs: {node_representations['id'][:5]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate node representation pkl file.")
    parser.add_argument('--csv', required=True, help='Path to the CSV file with metadata')
    parser.add_argument('--txt', required=True, help='Path to the .txt file with embeddings')
    parser.add_argument('--output', required=True, help='Path to output .pkl file')
    parser.add_argument('--full', action='store_true', help='Include gene/disease/pathway (for 4-entity version)')

    args = parser.parse_args()

    read_and_save_data(args.csv, args.txt, args.output, full=args.full)
