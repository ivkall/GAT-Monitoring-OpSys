import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from custom_gcn.custom_GCN import GCN, DataPoint
def standardize_data(data, mean=None, std=None):
    """
    Standardizes the data to have a mean of 0 and standard deviation of 1.
    If mean and std are provided, uses them to standardize the data.
    
    Parameters:
    data (np.array): The data to be standardized.
    mean (np.array): Optional. The mean to use for standardization.
    std (np.array): Optional. The standard deviation to use for standardization.

    Returns:
    np.array: The standardized data.
    """
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
    
    std[std == 0] = 1  # To avoid division by zero for constant features
    standardized_data = (data - mean) / std
    return standardized_data
# Initialize a GCN model
input_dim = 4
hidden_dim = 32  # You can tune this
hidden_dim2 = 16
output_dim = 12  # You can tune this, should match autoencoder input

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcn_model = GCN(input_dim, hidden_dim, hidden_dim2, output_dim).to(device)
gcn_model.eval()  # Set the model to evaluation mode

def extract_features(data, train = True):
    adj_matrix = torch.tensor(data[0].adjacency_matrix, dtype=torch.float).to(device)    

    gcn_transformed_features = []
    for i, data_point in enumerate(data):
    
        features_tensor = torch.tensor(data_point.features, dtype=torch.float).to(device)
        adj_matrix = torch.tensor(data_point.adjacency_matrix, dtype=torch.float).to(device)
        
        transformed_features = gcn_model(features_tensor, adj_matrix).detach().cpu().numpy()
        gcn_transformed_features.append(transformed_features.flatten())

    gcn_transformed_features = np.array(gcn_transformed_features)
    return gcn_transformed_features

def normalize_to_minus_one_and_one(data, train=True):
    min_vals = np.min(gcn_transformed_features, axis=0)
    max_vals = np.max(gcn_transformed_features, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges==0] = 0.00001
    
    normalized_data = (data - min_vals) / ranges  # Scale to [0, 1]
    normalized_data = normalized_data * 2 - 1  # Scale to [-1, 1]
    
    return normalized_data

def load_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

training_data = load_data("data1_100.pkl")
test_data_point = DataPoint(training_data[20].features, training_data[20].adjacency_matrix)
gcn_transformed_features = extract_features(training_data)

# Standardize the training data
#gcn_transformed_features_scaled = normalize_to_minus_one_and_one(gcn_transformed_features)
train_mean = np.mean(gcn_transformed_features, axis=0)
train_std = np.std(gcn_transformed_features, axis=0)
gcn_transformed_features_scaled=standardize_data (gcn_transformed_features)

train_data, val_data = train_test_split(gcn_transformed_features_scaled, test_size=0.2)

autoencoder = tf.keras.models.load_model('autoencoder_model')
with open('autoencoder_history.pkl', 'rb') as file:
    history = pickle.load(file)

# Evaluate the model and visualize the reconstruction quality
reconstructed_data = autoencoder.predict(val_data)

# Select random samples to display their original and reconstructed values
num_samples_to_display = 2
indices = np.random.choice(range(len(val_data)), num_samples_to_display)

print("Comparing original and reconstructed values for random data points:")
for i, index in enumerate(indices):
    print(f"\nData Point {i + 1} (Index {index}):")
    
    print(list(zip(val_data[index][:10], reconstructed_data[index][:10])))

# Visualizing the reconstruction for selected samples
train_loss = history.history['loss']
val_loss = history.history['val_loss']
print("Training Loss: ", train_loss[-1])
print("Validation Loss: ", val_loss[-1])

# # Plot the training and validation loss
# plt.figure(figsize=(10, 5))
# plt.plot(train_loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# Assuming the number of nodes and feature dimensions match the training data
num_nodes = 48
feature_dim = 4


# Create a random test data point
# test_data_point = create_test_data_point(num_nodes, feature_dim)
#test_data_point.features[0][0] = 123123
result_max = []
result_mse = []
for i in range(1,100,10):
    test_data_point.features[10][3] += 30*i
    test_data_point.features[4][2] += 30*i
    test_data_point.features[20][3] += 30*i

    test_data_gcn_features = extract_features([test_data_point], False)
    test_data_gcn_features_scaled = standardize_data (gcn_transformed_features,train_mean, train_std)
    test_data_prediction = autoencoder.predict(test_data_gcn_features_scaled)
    max_error = np.max(test_data_gcn_features_scaled - test_data_prediction)
    mse_error = mean_squared_error(test_data_gcn_features_scaled, test_data_prediction)

    result_max.append(max_error)
    result_mse.append(mse_error)
print("______________________________________________________________________")
print("______________________________________________________________________")
print(result_max)
print("______________________________________________________________________")
print(result_mse)
print("________________________________________________________")
print("________________________________________________________")

    

#print("(true, pred) values")
#print(list(zip(test_data_gcn_features_scaled[0], test_data_prediction[0]))[10:20])