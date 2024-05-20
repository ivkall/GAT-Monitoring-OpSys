import pandas as pd
import numpy as np
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error

class DataPoint:
    def __init__(self, features, adjacency_matrix):
        self.features = features
        self.adjacency_matrix = adjacency_matrix
    

#Save all nodes
df_full = pd.read_csv("data/test.csv")
unique_sources = set(df_full['Source'].unique())
unique_destinations = set(df_full['Destination'].unique())
unique_values_union = unique_sources.union(unique_destinations)
nodes = list(unique_values_union)
#Split data in the shiet.


# print(nodes, len(nodes))

data = []
total_len = len(df_full)
nbr_of_intervalls = 100
interval_length = total_len/nbr_of_intervalls

for i in range(0,nbr_of_intervalls):
    print(f"{i}/{nbr_of_intervalls}")
    df_limited = df_full[int(i*interval_length):int( (i+1)*interval_length) ]


    features = []

    for node in nodes:
        nbr_source = 0
        nbr_dest = 0
        avg_len_src = 0
        avg_len_dest = 0
        
        # Check if the node is in the first 100 rows for source
        if node in df_limited['Source'].values:


            
            nbr_source = df_limited['Source'].value_counts()[node]


            filtered_df = df_limited[df_limited['Source'] == node]
            avg_len_src = filtered_df['Length'].mean()
        
        
        # Check if the node is in the first 100 rows for destination
        if node in df_limited['Destination'].values:
            nbr_dest = df_limited['Destination'].value_counts()[node]

            filtered_df = df_limited[df_limited['Destination'] == node]
            avg_len_dest = filtered_df['Length'].mean()
            
        features.append([nbr_source, nbr_dest, avg_len_src, avg_len_dest])
    
    # print(features)

    limited_sources = np.array(df_limited['Source'].unique())
    limited_destinations = np.array(df_limited['Destination'].unique())

    adjacency_matrix = np.ones((len(nodes), len(nodes)))

    # Iterate through each row in the DataFrame and update the adjacency matrix
    for _, row in df_limited.iterrows():
        source_index = np.where(limited_sources == row['Source'])[0][0]
        destination_index = np.where(limited_destinations == row['Destination'])[0][0]
        adjacency_matrix[source_index, destination_index] = 1

    data_point = DataPoint(features, adjacency_matrix)
    data.append(data_point)


with open("data/data1_100_ones.pkl", "wb") as f:
    pickle.dump(data, f)