import pandas as pd
import numpy as np
import pickle
import os

class DataPoint:
    def __init__(self, features, adjacency_matrix):
        self.features = features  # Features is now a dictionary of nodes, each with a 'sent' and 'received' array
        self.adjacency_matrix = adjacency_matrix

script_dir = os.path.dirname(__file__)
csv_dir = '../data'
directory = os.path.join(script_dir, csv_dir)

all_nodes = []

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        df_full = pd.read_csv(f"data/{filename}")
        unique_sources = set(df_full['Source'].unique())
        unique_destinations = set(df_full['Destination'].unique())
        unique_nodes = unique_sources.union(unique_destinations)
        nodes = list(unique_nodes)
        print(filename, len(nodes), "nodes")
        # print(nodes)
        # print()
        for node in nodes:
            all_nodes.append(node)

all_nodes = list(set(all_nodes))
print(len(all_nodes), "total nodes")

# import utilities.statistic as stat
# stat.save_data("../data/all_nodes.pkl", all_nodes)