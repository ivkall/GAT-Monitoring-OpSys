import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models

class DataPoint:
    def __init__(self, features, adjacency_matrix):
        self.features = features
        self.adjacency_matrix = adjacency_matrix

# Load the data
with open("data2.pkl", "rb") as f:
    data = pickle.load(f)

# Flatten the features for each DataPoint and stack them into a training dataset
training_data = np.array([np.array(dp.features).flatten() for dp in data])

# Normalize the features
# It's important to avoid division by zero if std is zero for some features
mean = np.mean(training_data, axis=0)
std = np.std(training_data, axis=0)
std[std == 0] = 1  # Prevent division by zero
normalized_training_data = (training_data - mean) / std

# Define the autoencoder architecture
input_dim = normalized_training_data.shape[1]
encoding_dim = 32  # Increased encoding dimension

input_img = tf.keras.Input(shape=(input_dim,))
# Adding more layers and increasing the number of neurons
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
encoded = layers.Dense(encoding_dim // 2, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim // 4, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim // 8, activation='relu')(encoded)  # Additional layer
# You can add more layers or increase the neuron count in each layer

# For the decoding part, ensure to mirror the encoder
decoded = layers.Dense(encoding_dim // 8, activation='relu')(encoded)  # Corresponding additional layer
decoded = layers.Dense(encoding_dim // 4, activation='relu')(decoded)
decoded = layers.Dense(encoding_dim // 2, activation='relu')(decoded)
decoded = layers.Dense(encoding_dim, activation='relu')(decoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Training the more complex autoencoder
autoencoder.fit(normalized_training_data, normalized_training_data,
                epochs=100,  # You might want to adjust the number of epochs based on the training progression
                batch_size=100,  # Adjust batch size if necessary
                shuffle=True)

# Continue with the prediction steps as before