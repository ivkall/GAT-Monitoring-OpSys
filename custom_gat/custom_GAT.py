import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
import numpy as np
import utilities.statistic as stat

# Define your Graph Attention Network model
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList([
            GATConv(in_dim, hidden_dim, num_heads=num_heads),
            GATConv(hidden_dim * num_heads, out_dim, num_heads=1)
        ])

    # def forward(self, g, features):
    #     x = features
    #     for layer in self.layers:
    #         x = layer(g, x).flatten(1)
    #         x = F.elu(x)
    #     return x
    def forward(self, g, features):
        x = features
        for layer in self.layers[:-1]:  # Apply to all layers except the last
            x = layer(g, x)
            x = x.flatten(1) if x.dim() > 2 else x  # Only flatten if there are more than 2 dimensions
            x = F.elu(x)
        x = self.layers[-1](g, x).mean(dim=1)  # For the last layer, consider using mean or another operation to combine head outputs
        return x


def extract_features(gat_model, graph, features):
    gat_model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to compute gradients
        for layer in gat_model.layers:
            features = layer(graph, features).flatten(1)
            features = F.elu(features)
    return features


def preprocess_data(data_point):
    # Process each graph's adjacency matrix
    adjacency_matrix_np = np.array(data_point.adjacency_matrix)
    np.fill_diagonal(adjacency_matrix_np, 1)  # Add self-loops

    # Convert the adjacency matrix to COO format and then to a DGL graph
    graph = dgl.graph(np.nonzero(adjacency_matrix_np))

    # Normalize and convert node features to tensor
    node_features_np = np.array(data_point.features)
    node_features_tensor = torch.tensor(node_features_np, dtype=torch.float32)

    return (graph, node_features_tensor)


def train(gat_model, training_data, gat_epochs):
    graph, node_features_tensor = preprocess_data(training_data[0])

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error loss for unsupervised learning
    optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.001)

    train_loss = []
    # Training loop
    num_epochs = gat_epochs
    for epoch in range(num_epochs):
        logits = gat_model(graph, node_features_tensor)
        loss = criterion(logits, node_features_tensor)  # Reconstruction loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
        train_loss.append(loss.item())
    
    return gat_model


def train_gat_on_multiple_graphs(device, gat_model, graph_data_list, num_epochs, learning_rate=0.001, batch_size=32):
    gat_model.train()
    optimizer = torch.optim.Adam(gat_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Determine the number of batches per epoch
    num_batches = (len(graph_data_list) + batch_size - 1) // batch_size

    for epoch in range(num_epochs):
        total_loss = 0
        np.random.shuffle(graph_data_list)

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(graph_data_list))
            batch_data_points = graph_data_list[batch_start:batch_end]

            batch_graphs = []
            batch_features = []

            # Preprocess each graph and its features
            for data_point in batch_data_points:
                graph, features = preprocess_data(data_point)
                batch_graphs.append(graph)
                batch_features.append(features)

            batched_graph = dgl.batch(batch_graphs)
            batch_features = torch.cat(batch_features, dim=0)

            logits = gat_model(batched_graph, batch_features)
            loss = criterion(logits, batch_features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss}")

    return gat_model


def extract_all_features(gat_model, training_data):
    aggregated_features_list = []

    for data_point in training_data:
        graph, node_features_tensor = preprocess_data(data_point)

        # Use the trained GAT model to aggregate features for each graph
        aggregated_features = extract_features(gat_model, graph, node_features_tensor)
        aggregated_features_list.append(aggregated_features)

    # Concatenate all aggregated features from all graphs
    all_aggregated_features = torch.cat(aggregated_features_list, dim=0)

    

    return all_aggregated_features


def test_with_autoencoder(gat_model, train_mean, train_std, autoencoder, test_data_point):
    graph, node_features_tensor = preprocess_data(test_data_point)
    aggregated_features = extract_features(gat_model, graph, node_features_tensor)
    all_aggregated_features_scaled = stat.standardize_data(aggregated_features.detach().numpy(), train_mean, train_std)
    test_data = torch.tensor(all_aggregated_features_scaled, dtype=torch.float32)

    return autoencoder.evaluate(test_data.detach().numpy())