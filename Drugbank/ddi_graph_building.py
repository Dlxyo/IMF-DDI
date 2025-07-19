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

    part3 = torch.tensor(node_data['part3'], dtype=torch.float)

    def create_data(split_data):
       
        id1 = list(map(int, split_data['id1']))
        id2 = list(map(int, split_data['id2']))

        edge_index = torch.tensor([id1, id2], dtype=torch.long)

        edge_attr = split_data['ddi']
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        x = part3

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data

    train_data = create_data(edge_data['train'])
    val_data = create_data(edge_data['val'])
    test_data = create_data(edge_data['test'])


    return train_data, val_data, test_data




