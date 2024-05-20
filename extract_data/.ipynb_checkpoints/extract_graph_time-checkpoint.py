import pandas as pd
import numpy as np
import pickle

class DataPoint:
    def __init__(self, features, adjacency_matrix):
        self.features = features  # Features is now a dictionary of nodes, each with a 'sent' and 'received' array
        self.adjacency_matrix = adjacency_matrix

# Save all nodes
file_name = "kabelbrott_IO"
df_full = pd.read_csv(f"data/{file_name}.csv")
# unique_sources = set(df_full['Source'].unique())
# unique_destinations = set(df_full['Destination'].unique())
# unique_nodes = unique_sources.union(unique_destinations)
# nodes = list(unique_nodes)
# print(len(nodes), "nodes")
with open("data/all_nodes.pkl", "rb") as f:
        nodes = pickle.load(f)

# Initialize your data list
data = []
total_len = len(df_full)
nbr_of_intervals = 100
interval_length = total_len / nbr_of_intervals

total_time = df_full['Time'].iloc[-1]
interval_time = total_time / nbr_of_intervals
interval_index = []
i = 0

for j, row in df_full.iterrows():
    if row['Time'] >= interval_time * i:
        interval_index.append(j)
        i += 1


print(len(interval_index))


# Iterate over intervals
for i in range(nbr_of_intervals):
    print(f"{i}/{nbr_of_intervals}")
    df_limited = df_full[interval_index[i]:interval_index[i + 1]]

    # Initialize features dictionary with a sent and received array for each node
    features = {node: {'sent': np.zeros(len(nodes)), 'received': np.zeros(len(nodes))} for node in nodes}

    # Initialize adjacency matrix
    adjacency_matrix = np.ones((len(nodes), len(nodes)))

    # Populate features and adjacency matrix
    for _, row in df_limited.iterrows():
        source = row['Source']
        destination = row['Destination']
        length = row['Length']
        
        # Get indices of source and destination
        source_index = nodes.index(source)
        destination_index = nodes.index(destination)

        # Update sent and received packet counts
        features[source]['sent'][destination_index] += 1
        features[destination]['received'][source_index] += 1

        # Update adjacency matrix (if you still need this given the new features)
        adjacency_matrix[source_index, destination_index] = 1

    # Convert features to a more convenient list format
    features_list = [features[node]['sent'].tolist() + features[node]['received'].tolist() for node in nodes]

    # Create DataPoint
    data_point = DataPoint(features_list, adjacency_matrix)
    data.append(data_point)


# Save the data to a pickle file
with open(f"data/{file_name}_{nbr_of_intervals}_time-{interval_time:.1f}_ones.pkl", "wb") as f:
    pickle.dump(data, f)