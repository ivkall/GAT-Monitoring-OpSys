import pandas as pd
import numpy as np
import pickle

df_full = pd.read_csv("data/standard.csv")
# unique_sources = set(df_full['Source'].unique())
# unique_destinations = set(df_full['Destination'].unique())
# unique_nodes = unique_sources.union(unique_destinations)
with open("data/all_nodes.pkl", "rb") as f:
        nodes = pickle.load(f)
nodes = {node: i for i, node in enumerate(nodes)}

N = 1022652
lines = np.zeros((N,len(nodes)), dtype=int)
for i, row in df_full.iterrows():
    # print(row['Time'])
    source = row['Source']
    dest = row['Destination']
    lines[i, nodes[source]] = row['Length']
    lines[i, nodes[dest]] = -row['Length']
    print(i/N*100, "%")
    if i == N-1:
        break

# Parameter to define how many rows to sum into one
row_block_size = 10
# Calculate the new size of the array
new_size = N // row_block_size
# Initialize the new array
new_lines = np.zeros((new_size, len(nodes)), dtype=int)

# Sum the rows in blocks
for i in range(new_size):
    start_index = i* row_block_size
    end_index = start_index + row_block_size
    new_lines[i, :] = np.sum(lines[start_index:end_index, :], axis=0)
print(len(new_lines))
np.savetxt("machine-0-0.txt", new_lines, delimiter=",", fmt='%d')