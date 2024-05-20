import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error

class Autoencoder:
    def __init__(self, input_dim, encoding_dim, loaded_autoencoder=None):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        if loaded_autoencoder:
            self.autoencoder = loaded_autoencoder
        else:
            self.autoencoder = self.build_autoencoder()

    def build_autoencoder(self):
        input_img = tf.keras.Input(shape=(self.input_dim,))
        encoded = layers.Dense(self.encoding_dim, activation='tanh')(input_img)
        encoded = layers.Dropout(0.2)(encoded)  # Adding dropout with 20% rate
        encoded = layers.Dense(int(self.encoding_dim * 3/4), activation='tanh')(encoded)
        encoded = layers.Dropout(0.2)(encoded)  # Adding dropout
        encoded = layers.Dense(int(self.encoding_dim / 2), activation='tanh')(encoded)
        encoded = layers.Dropout(0.2)(encoded)  # Adding dropout

        decoded = layers.Dense(int(self.encoding_dim * 3/4), activation='tanh')(encoded)
        decoded = layers.Dropout(0.2)(decoded)  # Adding dropout
        decoded = layers.Dense(self.encoding_dim, activation='tanh')(decoded)
        decoded = layers.Dropout(0.2)(decoded)  # Adding dropout
        decoded = layers.Dense(self.input_dim)(decoded)

        autoencoder = models.Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        return autoencoder
    
    def evaluate(self, data):
        prediction = self.autoencoder.predict(data)
        max_error = np.max(data - prediction)
        mse_error = mean_squared_error(data, prediction)

        return prediction, max_error, mse_error
    
    def train_and_evaluate(self, train_data, val_data, epochs=200, batch_size=100):
        history = self.autoencoder.fit(train_data, train_data,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        validation_data=(val_data, val_data))
        reconstructed_data, _, _ = self.evaluate(val_data)
        print_evaluation(val_data, reconstructed_data, history)

        return history


def print_evaluation(val_data, reconstructed_data, history, num_samples_to_display=5):
    indices = np.random.choice(range(len(val_data)), num_samples_to_display)

    print("Comparing original and reconstructed values for random data points:")
    for i, index in enumerate(indices):
        print(f"\nData Point {i + 1} (Index {index}):")
        
        print(list(zip(val_data[index][:10], reconstructed_data[index][:10])))

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    print("Training Loss: ", train_loss[-1])
    print("Validation Loss: ", val_loss[-1])