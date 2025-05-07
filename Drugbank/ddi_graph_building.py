import pickle
import random
import torch
import networkx as nx
from torch_geometric.data import Data
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from collections import defaultdict
from collections import Counter

def load_data(node_file_path, edge_file_path):
    with open(node_file_path, 'rb') as f:
        node_data = pickle.load(f)

    ids = node_data['id']
    part3 = torch.tensor(node_data['part3'], dtype=torch.float)

    features = part3
    G = nx.Graph()
    id_to_index = {node_id: i for i, node_id in enumerate(ids)}

    for i, node_id in tqdm(enumerate(ids), desc="Adding Nodes"):
        G.add_node(node_id, enzyme=node_data['enzyme'][i], target=node_data['target'][i],
                   gene=node_data['gene'][i], disease=node_data['disease'][i])

    with open(edge_file_path, 'rb') as f:
        edge_data = pickle.load(f)
    
    edge_list = []
    edge_labels = []

    for i in range(len(edge_data['id1'])):
        id1 = edge_data['id1'][i]
        id2 = edge_data['id2'][i]
        ddi_id = int(edge_data['ddi_id'][i])

        if id1 in G.nodes and id2 in G.nodes:
            G.add_edge(id1, id2, ddi_id=ddi_id)
            edge_list.append((id_to_index[id1], id_to_index[id2]))
            edge_labels.append(ddi_id)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_labels, dtype=torch.float)

    pyg_data = Data(x=features, edge_index=edge_index, edge_attr=edge_attr)
    pyg_data.node_ids = ids
    pyg_data.node_data = node_data

    return G, pyg_data





def create_train_val_test_split(pyg_data, val_ratio=0.1, test_ratio=0.2, min_test_samples=1):
    from collections import defaultdict
    import random
    import torch
    from torch_geometric.data import Data

    edge_index = pyg_data.edge_index
    edge_attr = pyg_data.edge_attr


    class_edges = defaultdict(list)
    for i in range(edge_index.shape[1]):
        class_edges[edge_attr[i].item()].append(i)  

    train_edge_indices, train_edge_attrs = [], []
    val_edge_indices, val_edge_attrs = [], []
    test_edge_indices, test_edge_attrs = [], []


    train_class_counts = defaultdict(int)
    val_class_counts = defaultdict(int)
    test_class_counts = defaultdict(int)


    for class_idx, edges in class_edges.items():
        random.shuffle(edges)  
        num_edges = len(edges)

        num_test_samples = max(int(num_edges * test_ratio), min_test_samples)
        remaining = num_edges - num_test_samples


        num_val_samples = max(int(remaining * (val_ratio / (1 - test_ratio))), min_test_samples)
        num_train_samples = remaining - num_val_samples


        train_edges = edges[:num_train_samples]
        val_edges = edges[num_train_samples:num_train_samples + num_val_samples]
        test_edges = edges[num_train_samples + num_val_samples:]


        for edge_idx in train_edges:
            train_edge_indices.append([pyg_data.edge_index[0, edge_idx].item(),
                                       pyg_data.edge_index[1, edge_idx].item()])
            train_edge_attrs.append(edge_attr[edge_idx].item())
            train_class_counts[class_idx] += 1  


        for edge_idx in val_edges:
            val_edge_indices.append([pyg_data.edge_index[0, edge_idx].item(),
                                     pyg_data.edge_index[1, edge_idx].item()])
            val_edge_attrs.append(edge_attr[edge_idx].item())
            val_class_counts[class_idx] += 1 

        for edge_idx in test_edges:
            test_edge_indices.append([pyg_data.edge_index[0, edge_idx].item(),
                                      pyg_data.edge_index[1, edge_idx].item()])
            test_edge_attrs.append(edge_attr[edge_idx].item())
            test_class_counts[class_idx] += 1  

   
    train_edge_index = torch.tensor(train_edge_indices, dtype=torch.long).t().contiguous()
    train_edge_attr = torch.tensor(train_edge_attrs, dtype=torch.float)
    val_edge_index = torch.tensor(val_edge_indices, dtype=torch.long).t().contiguous()
    val_edge_attr = torch.tensor(val_edge_attrs, dtype=torch.float)
    test_edge_index = torch.tensor(test_edge_indices, dtype=torch.long).t().contiguous()
    test_edge_attr = torch.tensor(test_edge_attrs, dtype=torch.float)


    train_data = Data(x=pyg_data.x, edge_index=train_edge_index, edge_attr=train_edge_attr)
    val_data = Data(x=pyg_data.x, edge_index=val_edge_index, edge_attr=val_edge_attr)
    test_data = Data(x=pyg_data.x, edge_index=test_edge_index, edge_attr=test_edge_attr)


    return train_data, val_data, test_data


def print_node_info(node_index, pyg_data):
    node_id = pyg_data.node_ids[node_index]  
    features = pyg_data.x[node_index]  
    enzyme = pyg_data.node_data['enzyme'][node_index]  
    target = pyg_data.node_data['target'][node_index]  
    gene = pyg_data.node_data['gene'][node_index]  
    disease = pyg_data.node_data['disease'][node_index]  
    neighbors = pyg_data.edge_index[1][pyg_data.edge_index[0] == node_index].tolist()  

    print(f"Node ID: {node_id}")
    print(f"Enzyme: {enzyme}")
    print(f"Target: {target}")
    print(f"Gene: {gene}")
    print(f"Disease: {disease}")
    print(f"Features: {features}")
    print(f"Neighbors: {[pyg_data.node_ids[n] for n in neighbors]}")  


