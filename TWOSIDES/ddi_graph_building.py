import pickle
import torch
import networkx as nx
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.loader import DataLoader

import random
from collections import defaultdict
import pickle
import torch
from torch_geometric.data import Data
from imblearn.over_sampling import RandomOverSampler

def load_data(node_file_path, edge_file_path):
    with open(node_file_path, 'rb') as f:
        node_data = pickle.load(f)

    ids = node_data['id']
    part3 = torch.tensor(node_data['part3'], dtype=torch.float)

    features = part3
    G = nx.Graph()
    
    for i, node_id in tqdm(enumerate(ids), desc="Adding Nodes"):
        G.add_node(node_id, enzyme=node_data['enzyme'][i], target=node_data['target'][i],disease=node_data['disease'][i],gene=node_data['gene'][i])
    pyg_data = Data(x=features, node_ids = ids,node_data = node_data)

    return G, pyg_data


def load_edge_data(edge_file_path, node_file_path, max_ddi=200):
    with open(edge_file_path, 'rb') as f:
        edge_data = pickle.load(f)  

    with open(node_file_path, 'rb') as f:
        node_data = pickle.load(f)

    ids = node_data['id']
    part3 = torch.tensor(node_data['part3'], dtype=torch.float)


    id_to_index = {node_id: i for i, node_id in enumerate(ids)}

    def create_data(split_data):

        id1 = list(map(int, split_data['id1']))
        id2 = list(map(int, split_data['id2']))
        edge_index = torch.tensor([id1, id2], dtype=torch.long)

        edge_attr = torch.stack(split_data['ddi'])
        edge_polarity = list(map(int, split_data['polarity']))
 
        node_ids = torch.unique(edge_index.flatten())

        part3_values = [part3[id_to_index[node_id.item()]] for node_id in node_ids]
        x = torch.stack(part3_values)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_polarity = edge_polarity)

        return data

    train_data = create_data(edge_data['train'])
    val_data = create_data(edge_data['val'])
    val_data.x = train_data.x
    test_data = create_data(edge_data['test'])
    test_data.x = train_data.x
    return train_data, val_data, test_data


def load_edge_data_s1(edge_file_path, node_file_path, max_ddi=200):

    with open(edge_file_path, 'rb') as f:
        edge_data = pickle.load(f)  

    with open(node_file_path, 'rb') as f:
        node_data = pickle.load(f)

    ids = node_data['id']
    part3 = torch.tensor(node_data['part3'], dtype=torch.float)

    id_to_index = {node_id: i for i, node_id in enumerate(ids)}

    def create_data(split_data):

        id1 = list(map(int, split_data['id1']))
        id2 = list(map(int, split_data['id2']))

        edge_index = torch.tensor([id1, id2], dtype=torch.long)

        edge_attr = torch.stack(split_data['ddi'])
        edge_polarity = list(map(int, split_data['polarity']))

        node_ids = torch.unique(edge_index.flatten())

        part3_values = [part3[id_to_index[node_id.item()]] for node_id in node_ids]
        x = torch.stack(part3_values)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_polarity = edge_polarity)

        return data

    train_data = create_data(edge_data['train'])
    train_data.x = part3
    val_data = create_data(edge_data['val'])
    val_data.x = train_data.x
    test_data = create_data(edge_data['test'])
    test_data.x = train_data.x
    return train_data, val_data, test_data


def print_edge_info(edge_idx, data):

    num_edges = data.edge_index.size(1)
    if edge_idx < 0 or edge_idx >= num_edges:
        print(f"边索引 {edge_idx} 超出范围。总共有 {num_edges} 条边。")
        return

    src = data.edge_index[0, edge_idx].item()
    dst = data.edge_index[1, edge_idx].item()
    labels = data.edge_attr[edge_idx]
    polarity = data.edge_polarity[edge_idx]

    print(f"边 {edge_idx}: {src} -> {dst}, 标签: {labels}, 极性；{polarity}")

def count_labels(edge_attr):
    label_counts = torch.sum(edge_attr, dim=0).tolist()  
    return label_counts


