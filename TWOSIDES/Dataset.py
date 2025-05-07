import random
import torch
from torch_geometric.data import Data

def create_train_test_split(pyg_data, test_ratio=0.2):
    # Get the edge index and labels
    edge_index = pyg_data.edge_index
    edge_attr = pyg_data.edge_attr

    # Create a list of edges
    edge_list = [(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.shape[1])]
    
    # Shuffle and split edges
    random.shuffle(edge_list)
    split_index = int(len(edge_list) * (1 - test_ratio))
    
    train_edges = edge_list[:split_index]
    test_edges = edge_list[split_index:]

    # Create training graph data
    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
    train_edge_attr = torch.tensor([edge_attr[i].item() for i in range(len(edge_list)) if (edge_index[0, i].item(), edge_index[1, i].item()) in train_edges], dtype=torch.float)
    
    train_data = Data(x=pyg_data.x, edge_index=train_edge_index, edge_attr=train_edge_attr)

    # Create testing graph data
    test_edge_index = torch.tensor(test_edges, dtype=torch.long).t().contiguous()
    test_edge_attr = torch.tensor([edge_attr[i].item() for i in range(len(edge_list)) if (edge_index[0, i].item(), edge_index[1, i].item()) in test_edges], dtype=torch.float)

    test_data = Data(x=pyg_data.x, edge_index=test_edge_index, edge_attr=test_edge_attr)

    return train_data, test_data


