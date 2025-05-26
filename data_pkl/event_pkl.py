import pandas as pd
import pickle
import ast
import torch
from tqdm import tqdm

def read_and_save_multiple_datasets(train_csv, val_csv, test_csv, output_file, ddi_id_file):

    datasets = {}
    for split, csv_file in zip(['train', 'val', 'test'], [train_csv, val_csv, test_csv]):
        print(f"正在处理 {split} 数据集：{csv_file}")
        df = pd.read_csv(csv_file)

        id1 = df['id1'].tolist()
        id2 = df['id2'].tolist()
        ddi = df['ddi'].tolist()
        ddi_df = pd.read_csv(ddi_id_file, sep='\t')
        ddi_to_id = {row['ddi']: row['edge_index'] for _, row in ddi_df.iterrows()}

    # Map ddi_id to each entry in ddi
        ddi_id = [ddi_to_id[ddi_value] for ddi_value in ddi]

        datasets[split] = {
            'id1': id1,
            'id2': id2,
            'ddi': ddi,
            'ddi_id': ddi_id
        }
        
    # 保存为 pickle 文件
    with open(output_file, 'wb') as outfile:
        pickle.dump(datasets, outfile)

    print(f"数据已保存到 {output_file}")



if __name__ == "__main__":
    # 定义文件路径
    dataset_name = 'newdataset_event'
    train_csv_filepath = '/Data_scalability/train.csv'
    val_csv_filepath = '/Data_scalability/val.csv'
    test_csv_filepath = '/Data_scalability/test.csv'
    ddi_file = '/Data/durgbank_ddi_id.txt'
    save_path = f'/data/{dataset_name}.pkl'

    read_and_save_multiple_datasets(
        train_csv=train_csv_filepath,
        val_csv=val_csv_filepath,
        test_csv=test_csv_filepath,
        output_file=save_path,
        ddi_id_file=ddi_file
    )

