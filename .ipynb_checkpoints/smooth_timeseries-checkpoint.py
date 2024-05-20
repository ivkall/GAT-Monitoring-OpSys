import pandas as pd
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter

def aggregate_data(data, window_size=10):
    """Aggregate data using a sliding window approach, column-wise."""
    return np.array([np.sum(data[i:i+window_size, :], axis=0) for i in range(0, len(data) - window_size + 1, window_size)])

def apply_gaussian_bell(data, sigma=1):
    """Apply a Gaussian filter to each column separately."""
    return np.array([gaussian_filter(data[:, i], sigma=sigma) for i in range(data.shape[1])]).T

def moving_average(data, window_size=10):
    """Compute a moving average for each column separately."""
    return np.array([np.convolve(data[:, i], np.ones((window_size,))/window_size, mode='valid') for i in range(data.shape[1])]).T


df_full = pd.read_csv("data/noddöd_PLC.csv")
train_or_test = "test"


with open("data/all_nodes.pkl", "rb") as f:
    nodes = pickle.load(f)
nodes = {node: i for i, node in enumerate(nodes)}


N = len(df_full)
lines = np.zeros((N, len(nodes)), dtype=int)
for i, row in df_full.iterrows():
    source = row['Source']
    dest = row['Destination']
    lines[i, nodes[source]] = row['Length']
    lines[i, nodes[dest]] = -row['Length']


aggregated_lines = aggregate_data(lines, window_size=10)
gaussian_lines = apply_gaussian_bell(lines, sigma=5)
moving_avg_lines = moving_average(lines, window_size=10)
print("original", lines.shape)
print("aggregated", aggregated_lines.shape)
print("gaussian", gaussian_lines.shape)
print("moving avg", moving_avg_lines.shape)

# np.savetxt(f"../mtad-gat-pytorch/datasets/ServerMachineDataset/{train_or_test}/machine-0-0.txt", lines, delimiter=",", fmt='%d')
np.savetxt(f"../mtad-gat-pytorch/datasets/ServerMachineDataset/{train_or_test}/machine-0-0.txt", aggregated_lines, delimiter=",", fmt='%d')
# np.savetxt(f"../mtad-gat-pytorch/datasets/ServerMachineDataset/{train_or_test}/machine-0-0.txt", gaussian_lines, delimiter=",", fmt='%d')
# np.savetxt(f"../mtad-gat-pytorch/datasets/ServerMachineDataset/{train_or_test}/machine-0-0.txt", moving_avg_lines, delimiter=",", fmt='%d')