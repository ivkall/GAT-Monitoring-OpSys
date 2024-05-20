import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import torch.nn as nn
import  torch
from torch.utils.data import TensorDataset, DataLoader

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, num_layers=2, dropout_rate=0.2):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Initialize your encoder and decoder here with the provided parameters
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                               batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.decoder_hidden_dim = hidden_dim * 2  # Adjust for bidirectional
        self.decoder = nn.LSTM(self.decoder_hidden_dim, input_dim, num_layers=num_layers,
                               batch_first=True, dropout=dropout_rate)

    def forward(self, x):
        # Encoder
        _, (hidden, _) = self.encoder(x)

        # Adjust the hidden state for the decoder
        hidden = self._init_decoder_hidden(hidden)

        # Decoder
        # Ensure the decoder input is initialized to match the expected decoder's input size
        decoder_input = torch.zeros(x.size(0), self.seq_len, self.input_dim).to(x.device)
        # Reshape or transform hidden to match the expected hidden size for the decoder
        hidden = hidden.view(self.num_layers, -1, self.input_dim)
        decoded, _ = self.decoder(decoder_input, (hidden, torch.zeros_like(hidden)))

        return decoded
    def _init_decoder_hidden(self, hidden):
        # Process the hidden state from the bidirectional encoder for the decoder
        if hidden.size(0) == self.num_layers * 2:
            # Reshape hidden state to combine the bidirectional layers
            hidden = hidden.view(self.num_layers, 2, hidden.size(1), self.hidden_dim)
            hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        hidden = hidden.transpose(0, 1).contiguous()

        # Ensure hidden state size matches decoder expectations
        hidden = hidden.view(self.num_layers, -1, self.decoder_hidden_dim)  # Correct the shape for the decoder
        return hidden
    
data = np.genfromtxt("NODE_SERIES2.txt", delimiter=',')[:10000,:]
print(data.shape)
input_dim = data.shape[1]
hidden_dim = 24  # You can change this
time_steps = 10  # The number of time steps you want to use for predictions
seq_len = time_steps
features = 48  # The number of features/columns in your dataset

sequence_length = 100

X = []
Y = []

# Construct sequences and the next timestep as the target
for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length, :])  # Sequence of 10 data points
    Y.append(data[i+sequence_length, :])  # The following data point

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)

# Create DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
num_layers = 2  # or any other number you prefer
dropout_rate = 0.2  # or any other value
model = LSTMAutoencoder(input_dim, hidden_dim, seq_len, num_layers, dropout_rate)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, Y_batch in data_loader:
        optimizer.zero_grad()

        # Forward pass through the model
        output = model(X_batch)

        # Adjusting the output dimension for comparison:
        # We compare the last timestep's prediction from the output with the target
        output = output[:, -1, :].squeeze()  # Squeeze to remove the middle dimension, making it [32, 48]

        # Ensuring output and Y_batch now have the same shape for loss calculation
        assert output.shape == Y_batch.shape, f"Output shape {output.shape} does not match target shape {Y_batch.shape}"

        loss = criterion(output, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {total_loss/len(data_loader)}')



