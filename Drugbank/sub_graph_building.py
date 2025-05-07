import networkx as nx
from Dataset import create_train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import os




def build_graph_from_nodes(node_ids, pyg_data, target_key, data_type):
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    
    id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
    edge_indices = []
    target_map = {}
    
   
    processed_node_data = []
    for i, node_id in tqdm(enumerate(node_ids), desc=f"Building {target_key} map"):
        node_data = pyg_data.node_data[target_key][i]
        if not isinstance(node_data, str):
            node_targets = set()  
        else:
            node_targets = set(node_data.split('|'))  
        processed_node_data.append(node_targets)
        
        for target in node_targets:
            if target not in target_map:
                target_map[target] = []
            target_map[target].append(node_id)

    n = 1
    max_iterations =5
    while n <= max_iterations:
        edge_count = 0
        current_G = nx.Graph()
        current_G.add_nodes_from(node_ids)

        for node_list in tqdm(target_map.values(), desc=f"Adding edges for n={n}"):
            if len(node_list) > 1:
                for i in range(len(node_list)):
                    for j in range(i + 1, len(node_list)):
                        
                        node_i_targets = processed_node_data[id_to_index[node_list[i]]]
                        node_j_targets = processed_node_data[id_to_index[node_list[j]]]

                        common_targets = len(node_i_targets & node_j_targets)  

                        if common_targets >= n and not current_G.has_edge(node_list[i], node_list[j]):
                            current_G.add_edge(node_list[i], node_list[j], weight=1)
                            edge_count += 1
        
        if edge_count == 0:
            print(f"No more edges added with {n} or more common {target_key}, stopping.")
            break
        print(f"Iteration {n}: Total edges added = {current_G.number_of_edges()}") 

        edge_list = [(id_to_index[u], id_to_index[v]) for u, v in current_G.edges()]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_indices.append(edge_index)
        
        n += 1

    return edge_indices
