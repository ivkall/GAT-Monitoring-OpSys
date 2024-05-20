import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import utilities.statistic as stat

# Define the Graph Attention Layer

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.matmul(adj, x)
        return x

# Define the GCN Model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim1)
        # Additional intermediate layer
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2)
        self.gc3 = GraphConvolution(hidden_dim2, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))  # Pass through the second (new) layer with ReLU
        x = self.gc3(x, adj)  # Output layer does not have ReLU if it's the final output
        return x

    
# Load and prepare the data
class DataPoint:
    def __init__(self, features, adjacency_matrix):
        self.features = features
        self.adjacency_matrix = adjacency_matrix


def extract_features(device, gcn_model, data):
    adj_matrix = torch.tensor(data[0].adjacency_matrix, dtype=torch.float).to(device)    

    gcn_transformed_features = []
    for i, data_point in enumerate(data):
    
        features_tensor = torch.tensor(data_point.features, dtype=torch.float).to(device)
        adj_matrix = torch.tensor(data_point.adjacency_matrix, dtype=torch.float).to(device)
        
        transformed_features = gcn_model(features_tensor, adj_matrix).detach().cpu().numpy()
        gcn_transformed_features.append(transformed_features.flatten())

    gcn_transformed_features = np.array(gcn_transformed_features)
    return gcn_transformed_features


def test_with_autoencoder(device, gcn_model, datapoint, train_mean, train_std, autoencoder):
    test_data_gcn_features = extract_features(device, gcn_model, datapoint)
    test_data_gcn_features_scaled = stat.standardize_data(test_data_gcn_features, train_mean, train_std)

    return autoencoder.evaluate(test_data_gcn_features_scaled)