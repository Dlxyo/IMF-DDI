
import pandas as pd
import pickle

def read_and_save_data_4entity(txt_file3, csv_file, output_file):
    def read_float_array(file_path):
        with open(file_path, 'r') as f:
            return [list(map(float, line.split())) for line in f.readlines()]

    data3 = read_float_array(txt_file3)

    df = pd.read_csv(csv_file)
    
    id = df['id'].tolist()
    enzymes = df['enzyme'].tolist()
    targets = df['target'].tolist()
    genes = df['gene'].tolist()
    diseases = df['disease'].tolist()
    
    node_representations = {
        'id':id,        
        'part3': data3,
        'enzyme': enzymes,
        'target': targets,
        'gene': genes,
        'disease': diseases
    }

    # Save as pkl file
    with open(output_file, 'wb') as outfile:
        pickle.dump(node_representations, outfile)

def read_and_save_data(txt_file3, csv_file, output_file):
    def read_float_array(file_path):
        with open(file_path, 'r') as f:
            return [list(map(float, line.split())) for line in f.readlines()]

    data3 = read_float_array(txt_file3)

    df = pd.read_csv(csv_file)
    
    id = df['id'].tolist()
    enzymes = df['enzyme'].tolist()
    targets = df['target'].tolist()

    
    node_representations = {
        'id':id,        
        'part3': data3,
        'enzyme': enzymes,
        'target': targets,
    }

    # Save as pkl file
    with open(output_file, 'wb') as outfile:
        pickle.dump(node_representations, outfile)



# Define file paths
dataset_name = 'new_dataset'
dataset_csv_filepath = '/Data_scalability/new_drug_allentity.csv'
part3_filepath = 'embedding/new_dataset_embeddings.txt'
save_path = f'/data/{dataset_name}.pkl'

read_and_save_data_4entity(part3_filepath, dataset_csv_filepath, save_path)
