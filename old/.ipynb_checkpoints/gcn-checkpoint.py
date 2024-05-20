import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DataPoint:
    def __init__(self, features, adjacency_matrix):
        self.features = features
        self.adjacency_matrix = adjacency_matrix

# Load the data
with open("data2_100.pkl", "rb") as f:
    data = pickle.load(f)



class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = self.linear(x).float()  # Ensuring the output is float
        x = torch.matmul(adj.float(), x)  # Ensuring both operands are float
        return x


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)  # Second layer
        self.gc3 = GraphConvolution(hidden_dim, hidden_dim)  # Second layer
        self.gc4 = GraphConvolution(hidden_dim, hidden_dim)  # Fourth layer
        self.gc5 = GraphConvolution(hidden_dim, output_dim)  # Output layer with output_dim neurons


    def forward(self, x, adj):
    
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = self.gc5(x, adj)  # No ReLU here before the final softmax
        return F.log_softmax(x, dim=1)

# Example usage
num_nodes = len(data[0].features)  # Number of nodes in the graph
num_features = 4  # Number of features per node
hidden_channels = 8  # Hidden layer size
output_channels = 8 # Number of output channels for the final features


transformed_features = []
num_samples = 10000
sequence_length = 20  # Each data point has 20 values

# Generate sinusoidal data
x_values = np.linspace(0, 100*np.pi, num_samples * sequence_length)
sinus_data = np.sin(x_values).reshape(num_samples, sequence_length)

# Create DataPoint instances without adjacency matrices
data = [DataPoint(features=sinus_data[i], adjacency_matrix=None) for i in range(num_samples)]
for data_point in data:
    # Ensure data_point is a NumPy array of a specific type, e.g., float32.
    temp = np.array(data_point.features, dtype=np.float).flatten()
    # Convert the flattened array to a PyTorch tensor.
    temp_tensor = torch.tensor(temp, dtype=torch.float)
    

    transformed_features.append(temp_tensor)


all_transformed_features = torch.stack(transformed_features, dim=0)

# Convert to numpy for normalization
all_transformed_features_numpy = all_transformed_features.numpy()

# Normalize the data
mean = np.mean(all_transformed_features_numpy, axis=0)
std = np.std(all_transformed_features_numpy, axis=0)
std[std == 0] = 1  # Avoid division by zero for any standard deviation that is 0
normalized_training_data_numpy = (all_transformed_features_numpy - mean) / std

# Convert the normalized data back to a TensorFlow tensor for the autoencoder
normalized_training_data = tf.convert_to_tensor(normalized_training_data_numpy, dtype=tf.float32)




# Defining the autoencoder architecture
# Ensure input_dim matches the dimension of the GCN output



# Define the autoencoder architecture
input_dim = normalized_training_data.shape[1] 
encoding_dim =  normalized_training_data.shape[1]*2  # Increased encoding dimension

input_img = tf.keras.Input(shape=(input_dim,))
# Adding more layers and increasing the number of neurons
encoded = layers.Dense(encoding_dim, activation='tanh')(input_img)
encoded = layers.Dropout(0.2)(encoded)  # Adding dropout for regularization
encoded = layers.Dense(encoding_dim * 4, activation='tanh')(encoded)
encoded = layers.Dense(encoding_dim, activation='tanh')(encoded)
encoded = layers.Dense(encoding_dim * 2, activation='tanh')(encoded)
encoded = layers.Dense(encoding_dim , activation='tanh')(encoded)

# Decoder: Mirroring the encoder structure
decoded = layers.Dense(encoding_dim , activation='tanh')(encoded)
decoded = layers.Dense(encoding_dim * 2, activation='tanh')(decoded)
decoded = layers.Dense(encoding_dim, activation='tanh')(decoded)
decoded = layers.Dense(encoding_dim * 4, activation='tanh')(decoded)
decoded = layers.Dropout(0.2)(decoded)  # Adding dropout for regularization
decoded = layers.Dense(input_dim, activation='tanh')(decoded)  # Using sigmoid for the final layer

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
#SOME RANDOM FLIPPING 
normalized_training_data = normalized_training_data[1:]

normalized_training_data_np = normalized_training_data.numpy()
train_data_np, val_data_np = train_test_split(normalized_training_data_np, test_size=0.2)
train_data = tf.convert_to_tensor(train_data_np, dtype=tf.float32)
val_data = tf.convert_to_tensor(val_data_np, dtype=tf.float32)
# Now, train your model using the training data and validate using the validation data
history = autoencoder.fit(train_data, train_data,
                          epochs=100,
                          batch_size=100,
                          shuffle=True,
                          validation_data=(val_data, val_data))  # Use the held-out validation data here

test_data = normalized_training_data[:4]  # Just an example to get 10 samples


test_data_np = test_data.numpy() if isinstance(test_data, tf.Tensor) else test_data

# Convert bruh to a NumPy array and append it to test_data_np



# If you need test_data as a TensorFlow tensor for further processing
test_data_tensor = tf.convert_to_tensor(test_data_np, dtype=tf.float32)
# Convert test data to TensorFlow tensor


# Get the reconstructed outputs
reconstructed_data = autoencoder.predict(test_data_tensor)

# Compare the original data with the reconstructed data
# This step is more about evaluation rather than prediction
# You can use various metrics like MSE or visualize the differences
mse_values = np.mean(np.power(test_data - reconstructed_data, 2), axis=1)

# Find the index of the sample with the smallest MSE
max_mse_index = np.argmax(mse_values)


sorted_indices = np.argsort(mse_values)


# Step 2: Print the four smallest MSEs and corresponding data
print("Four biggest MSEs and their data points:")
for index in sorted_indices:
    print(f"\nMSE {mse_values[index]} at index {index}:")
    print("Original Data Point:")
    print(test_data[index])
    print("Reconstructed Data Point:")
    print(reconstructed_data[index])
# print("___________________________________________")
# print("MANUAL TEST")
# print("___________________________________________")
# print("___________________________________________")
# print("MANUAL TEST")
# print("___________________________________________")
# experimental_data_point = np.array([0.09261464, 0.096355848, 0.69388162, 0.69411874 ,0.69414674 ,0.60065913,
#  0.09296079 ,0.09356105])
# print("Test data before reconstruct:", experimental_data_point)
# # Normalize the experimental data point


# # Convert the normalized data point to a TensorFlow tensor for the autoencoder
# experimental_data_tensor = tf.convert_to_tensor(experimental_data_point.reshape(1, -1), dtype=tf.float32)

# # Use the autoencoder to reconstruct the data
# reconstructed_experimental_data = autoencoder.predict(experimental_data_tensor)

# # Output the reconstructed data and calculate MSE
# print("Reconstructed Data:", reconstructed_experimental_data.flatten())

# mse = np.mean(np.power(experimental_data_tensor - reconstructed_experimental_data, 2), axis=1)
# print("MSE:", mse)