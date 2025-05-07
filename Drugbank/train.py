import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim
from ddi_graph_building import *
from sub_graph_building import build_graph_from_nodes
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import numpy as np
from torch.nn import LayerNorm

class TransformerHead(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super(TransformerHead, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.norm1 = LayerNorm(hidden_channels)
        self.norm2 = LayerNorm(out_channels)

    def forward(self, x, edge_index):
        residual = x
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.norm1(x + residual)  # Normalized residual connection
        x = self.conv2(x, edge_index)
        return F.leaky_relu(self.norm2(x + residual), negative_slope=0.01)

class ECALayer(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)  
        y = y.view(b, 1, c)  
        y = self.conv(y)  
        y = self.sigmoid(y)  
        y = y.view(b, c, 1, 1)  

        return x * y.expand_as(x)

class MultiGraphWeightedGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_classes, embed_dim, num_heads, num_encoder_layers, selected_features, dropout=0.1):
        super(MultiGraphWeightedGCN, self).__init__()

        self.target_gcn = GCN(in_channels, hidden_channels, out_channels)
        self.enzyme_gcn = GCN(in_channels, hidden_channels, out_channels)
        self.gene_gcn = GCN(in_channels, hidden_channels, out_channels)
        self.disease_gcn = GCN(in_channels, hidden_channels, out_channels)

        self.layer_weights = nn.ParameterDict({
            feature: nn.Parameter(torch.ones(num_layers, len(selected_features))) for feature in selected_features
        })

        self.retain_weights = nn.ParameterDict({
            feature: nn.Parameter(torch.ones(num_layers)) for feature in selected_features
        })

        self.selected_features = selected_features
        self.transformer_head = TransformerHead(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_encoder_layers, num_classes=num_classes, dropout=dropout)
        self.fc_raw = nn.Linear(out_channels * len(selected_features), 32)
        self.fc_dim = nn.Linear(in_channels, in_channels)
        self.fc_final = nn.Linear(out_channels +32, in_channels)

    def compute_combined_output(self, x_target, x_enzyme, x_gene, x_disease, layer_weights):
        layer_weights_softmax = F.softmax(layer_weights, dim=0)
        combined = (
            layer_weights_softmax[0] * x_target +
            layer_weights_softmax[1] * x_enzyme +
            layer_weights_softmax[2] * x_gene +
            layer_weights_softmax[3] * x_disease
        )
        return combined

    def forward(self, x, edge_indices_dict):
        x_target = x
        x_enzyme = x
        x_gene = x
        x_disease = x

        for i in range(len(self.retain_weights[self.selected_features[0]])):
            x_target = self.target_gcn(x_target, edge_indices_dict['target'][i])
            x_enzyme = self.enzyme_gcn(x_enzyme, edge_indices_dict['enzyme'][i])
            x_gene = self.gene_gcn(x_gene, edge_indices_dict['gene'][i])
            x_disease = self.disease_gcn(x_disease, edge_indices_dict['disease'][i])

            combined_output_target = self.compute_combined_output(x_target, x_enzyme, x_gene, x_disease, self.layer_weights['target'][i])
            combined_output_enzyme = self.compute_combined_output(x_target, x_enzyme, x_gene, x_disease, self.layer_weights['enzyme'][i])
            combined_output_gene = self.compute_combined_output(x_target, x_enzyme, x_gene, x_disease, self.layer_weights['gene'][i])
            combined_output_disease = self.compute_combined_output(x_target, x_enzyme, x_gene, x_disease, self.layer_weights['disease'][i])

            retain_weights_target = torch.clamp(self.retain_weights['target'][i], min=1e-5, max=1)
            retain_weights_enzyme = torch.clamp(self.retain_weights['enzyme'][i], min=1e-5, max=1)
            retain_weights_gene = torch.clamp(self.retain_weights['gene'][i], min=1e-5, max=1)
            retain_weights_disease = torch.clamp(self.retain_weights['disease'][i], min=1e-5, max=1)

            x_target = retain_weights_target * x_target + (1 - retain_weights_target) * combined_output_target
            x_enzyme = retain_weights_enzyme * x_enzyme + (1 - retain_weights_enzyme) * combined_output_enzyme
            x_gene = retain_weights_gene * x_gene + (1 - retain_weights_gene) * combined_output_gene
            x_disease = retain_weights_disease * x_disease + (1 - retain_weights_disease) * combined_output_disease
        
        
        x_target = x_target.unsqueeze(0)
        x_enzyme = x_enzyme.unsqueeze(0)
        x_gene = x_gene.unsqueeze(0)
        x_disease = x_disease.unsqueeze(0)


        
        raw_concatenated = torch.cat([x_target, x_enzyme, x_gene, x_disease], dim=2).squeeze(0)  
        _, feature_dim = raw_concatenated.shape
        self_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8, dropout=0.1, batch_first=True).to(device)
        attn_output, _ = self_attention(raw_concatenated.unsqueeze(0), raw_concatenated.unsqueeze(0), raw_concatenated.unsqueeze(0))
        raw_concatenated = raw_concatenated * attn_output.squeeze(0)



        Dimensional_concatenated = torch.cat([x_target, x_enzyme, x_gene, x_disease], dim=0)
        Dimensional_concatenated = Dimensional_concatenated.unsqueeze(0)
        channel_attention = ChannelAttention(Dimensional_concatenated.size(1)).to(device)
        Dimensional_concatenated = channel_attention(Dimensional_concatenated)
        Dimensional_concatenated = Dimensional_concatenated.squeeze(0)
        Dimensional_concatenated = Dimensional_concatenated.mean(dim=0)


        raw_concatenated = self.fc_raw(raw_concatenated)

        final_representation = self.fc_final(torch.cat([raw_concatenated, Dimensional_concatenated], dim=1))

        return final_representation

def train_model(train_data, model, optimizer, criterion, device, edge_indices_dict, alpha=1.0, beta=1.0, noise_std=5e-3, lambda_reg=1e-2):

    model.train()
    optimizer.zero_grad()

    x = train_data.x.to(device)

    # Forward pass with original inputs
    output_original = model(x, {key: [ei.to(device) for ei in edge_indices] for key, edge_indices in edge_indices_dict.items()})

    node1_embeds = output_original[train_data.edge_index[0]] 
    node2_embeds = output_original[train_data.edge_index[1]]  
    edge_representations_original = node1_embeds - node2_embeds  
    preds_original = model.transformer_head(edge_representations_original.unsqueeze(0))

    labels = train_data.edge_attr.to(device).long()
    loss1 = criterion(preds_original.squeeze(0), labels)
    loss1.backward()
    optimizer.step()
    return loss1.item()


def evaluate_model(test_data, model, device, edge_indices_dict):
    model.eval()
    true_labels = []
    predicted_labels = []
    predicted_probs = []

    with torch.no_grad():
        x = test_data.x.to(device)

        x = model(x, {key: [ei.to(device) for ei in edge_indices] for key, edge_indices in edge_indices_dict.items()})

        node1_embeds = x[test_data.edge_index[0]]  # Shape: [num_edges, embedding_dim]
        node2_embeds = x[test_data.edge_index[1]]  # Shape: [num_edges, embedding_dim]
        edge_representations = node1_embeds - node2_embeds  # Shape: [num_edges, embedding_dim]

        preds = model.transformer_head(edge_representations.unsqueeze(0))

        squeezed_preds = preds.squeeze(0)  # Shape: (num_edges, num_classes)

        pred_prob = F.softmax(squeezed_preds, dim=1)  # Shape: (num_edges, num_classes)
        pred_label = squeezed_preds.argmax(dim=1)  # Shape: (num_edges,)
        true_label = test_data.edge_attr  # Assuming shape: (num_edges,)
        pred_prob_np = pred_prob.cpu().numpy()  # Shape: (num_edges, num_classes)
        predicted_labels = pred_label.tolist()
        true_labels = true_label.tolist()
        predicted_probs = pred_prob_np.tolist()

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    predicted_probs = np.array(predicted_probs)

    acc = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    return acc, f1, kappa


def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(47)
    opt = 'test'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log_file_path = 'knowdddi.txt'  # You can change the file path as needed
    best_model_path = "knowdddi.pth"
    node_file_path = './Drugbank/pkl/drugbank_drug.pkl'
    edge_file_path = './Drugbank/pkl/drugbank_knowddi.pkl'
    h
    G, pyg_data = load_data(node_file_path, edge_file_path)
    train_data, val_data, test_data = load_edge_data(edge_file_path,node_file_path)
    node_ids = list(range(train_data.x.size(0)))
    selected_features = ['target', 'enzyme', 'gene', 'disease']  # Can dynamically modify this list


    edge_indices_dict = {
        feature: build_graph_from_nodes(node_ids, pyg_data, feature, 'train') for feature in selected_features
    }
    

    # Initialize model and optimizer
    num_features = train_data.x.size(1)  # Input feature dimension
    hidden_channels = 512
    out_channels = 512
    embed_dim = out_channels  # Transformer input dimension should match GCN output dimension
    num_heads = 8  # Number of transformer heads
    num_encoder_layers = 2  # Number of transformer encoder layers
    num_classes = 86  # Number of classification classes

    # Initialize the model
    model = MultiGraphWeightedGCN(
        in_channels=num_features,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=len(edge_indices_dict[selected_features[0]]),  # Assuming number of layers matches edge index length
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        selected_features=selected_features
    ).to(device)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adjust learning rate as needed
    criterion = nn.CrossEntropyLoss()

    # Move data to device
    train_data.x = train_data.x.to(device)
    train_data.edge_index = train_data.edge_index.to(device)
    train_data.edge_attr = train_data.edge_attr.to(device)

    test_data.x = test_data.x.to(device)
    test_data.edge_index = test_data.edge_index.to(device)
    test_data.edge_attr = test_data.edge_attr.to(device)

    val_data.x = val_data.x.to(device)
    val_data.edge_index = val_data.edge_index.to(device)
    val_data.edge_attr = val_data.edge_attr.to(device)

    
    if opt=='test':
        opt_data = test_data
    elif opt =='val':
        opt_data = val_data
    # Training and evaluation
    best_acc = 0
    best_metrics = {}
    epochs = 10000
    
    with open(log_file_path, 'a', buffering=1) as log_file:  
        log_file.write("Epoch,Train Loss,ACC,F1,Cohen's Kappa\n")  # Header
        for epoch in range(epochs):
            # Train the model with input perturbations and regularization
            train_loss = train_model(
                train_data,
                model,
                optimizer,
                criterion,
                device,
                edge_indices_dict,
                alpha=1.0,
                beta=1.0,
                noise_std=1e-3,      # Standard deviation for input noise
                lambda_reg=1e-2      # Regularization coefficient
            )

            # Evaluate the model on the test set
            
            acc, f1, kappa = evaluate_model(opt_data, model, device, edge_indices_dict)

            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"Train Loss: {train_loss:.4f}, ACC: {acc:.4f}, F1: {f1:.4f}, Cohen's Kappa: {kappa:.4f}")
            log_file.write(f"{epoch + 1},{train_loss:.4f},{acc:.4f},{f1:.4f},{kappa:.4f}\n")
            log_file.flush()  
            # Update the best results
            if acc > best_acc:
                best_acc = acc
                best_metrics = {
                    'epoch': epoch + 1,
                    'loss': train_loss,
                    'acc': acc,
                    'f1': f1,
                    'kappa': kappa,
                }
                torch.save(model.state_dict(), best_model_path) 
            if (epoch + 1) % 10 == 0:
                print("Best Results:")
                print(f"Epoch: {best_metrics['epoch']}, Train Loss: {best_metrics['loss']:.4f}, "
                    f"ACC: {best_metrics['acc']:.4f}, F1: {best_metrics['f1']:.4f}, Cohen's Kappa: {best_metrics['kappa']:.4f}")
                log_file.write(f"Best up to Epoch {epoch + 1}: Epoch {best_metrics['epoch']}, "
                                f"Train Loss: {best_metrics['loss']:.4f}, ACC: {best_metrics['acc']:.4f}, "
                                f"F1: {best_metrics['f1']:.4f}, Cohen's Kappa: {best_metrics['kappa']:.4f}\n")
                log_file.flush()  