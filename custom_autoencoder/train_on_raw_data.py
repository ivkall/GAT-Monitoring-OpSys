import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
def hex_to_normalized_int(hex_str):
    return int(hex_str, 16) / 255.0

def process_line(line):
    # Extracting hexadecimal values; this may need adjustment based on the exact file structure
    hex_values = line.split('|')[2:]  # Split and ignore the initial part of the line
    hex_values = [value.strip() for value in hex_values if value.strip()]  # Remove empty strings and strip spaces
    return [hex_to_normalized_int(value) for value in hex_values]

def read_starting_from_line_and_every_nth_line(file_path, start_line, step):
    lines = []
    with open(file_path, 'r') as file:
        for index, line in enumerate(file):
            if index < 1000000 and index >= start_line - 1 and (index - start_line + 1) % step == 0:
                if index >= 100000:
                            
                    break
                print(index)
                processed_line = process_line(line)
                lines.append(processed_line)
    return lines

load = False
if not load:
    file_path = 'data/standard.txt'  # Replace 'your_file.txt' with the path to your text file
    start_line = 3
    step = 4
    result = read_starting_from_line_and_every_nth_line(file_path, start_line, step)

    with open("data/cap_1_array.pkl", "wb") as f:
        pickle.dump(result, f)
else:
    with open("data/cap_1_array.pkl", "rb") as f:
        result = pickle.load(f)


max_length = max(len(item) for item in result)

load = False
if not load:
    # Initialize an empty list to hold the padded data
    padded_data = []

    # Pad each array and print the progress
    for index, item in enumerate(result):
        padded_item = item + [0] * (max_length - len(item))
        padded_data.append(padded_item)
        
        # Print the progress
        progress = (index + 1) / len(result) * 100
        print(f"Progress: {progress:.2f}%")

    # Convert the list to a NumPy array for consistency in data structure
    # padded_data = np.array(padded_data)
    # test
    # Show the results
    print("First 5 elements of the padded first item:", padded_data[0])
  
    print("Max length:", max_length)
    
    with open("data/cap_1_array_padded.pkl", "wb") as f:
        pickle.dump(padded_data, f)
else:
    with open("data/cap_1_array_padded.pkl", "rb") as f:
        padded_data = pickle.load(f)


padded_data_numpy = np.array(padded_data)
train_data, val_data = train_test_split(padded_data_numpy, test_size=0.2)
input_dim = max_length  # max_length from your padded data
print("STARTING TRAINNING")
# Define the model


# Assume input_dim is defined as the length of your padded data
input_dim = max_length  # max_length from your padded data

# Define encoder layers
input_layer = Input(shape=(input_dim,))
encoder = Dense(128, activation='relu')(input_layer)
encoder = Dense(64, activation='relu')(encoder)
encoder = Dense(32, activation='relu')(encoder)  # Encoded representation

# Define decoder layers (mirror the encoder)
decoder = Dense(64, activation='relu')(encoder)
decoder = Dense(128, activation='relu')(decoder)
decoder_output = Dense(input_dim, activation='sigmoid')(decoder)  # Reconstruction of input

# Create autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder_output)
autoencoder.compile(optimizer='adam', loss='mse')

# Summary of the autoencoder architecture
autoencoder.summary()

history = autoencoder.fit(train_data, train_data,
                            epochs=100,
                            batch_size=256,
                            shuffle=True,
                            validation_data=(val_data, val_data)
                            )
import numpy as np
import random

def modify_data(data_point):
    modified_data = data_point.copy()
    for i in range(2):
        nbr1 = random.randint(1, 1000) / 1000.0
        nbr2 = random.randint(0, len(modified_data) - 1)
        modified_data[nbr2] = nbr1
    return modified_data

# Select 100 random indices from the training data
random_indices = np.random.randint(0, len(train_data), size=200)

evaluation_results = []

for idx in random_indices:
    original_data = train_data[idx]
    modified_data = modify_data(original_data)
    modified_data_reshaped = np.array([modified_data])
    evaluation = autoencoder.evaluate(modified_data_reshaped, modified_data_reshaped, verbose=0)
    evaluation_results.append(evaluation)

mean_random_error = np.mean(evaluation_results)
print(f"Mean random error over 100 evaluations: {mean_random_error}")