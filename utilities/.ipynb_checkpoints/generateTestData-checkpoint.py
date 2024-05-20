import numpy as np

class DataPoint:
    """
    Represents a data point in the network, holding node features and the adjacency matrix.
    """
    
    def __init__(self, features, adjacency_matrix):
        self.features = features
        self.adjacency_matrix = adjacency_matrix
 
def generateData(datapoint, nbr_of_generations, type_of_error, divide_by, x_interval):
    """
    Generates data by simulating different types of errors in the network.

    :param datapoint: The original datapoint to base simulations on.
    :param nbr_of_generations: The number of data points to generate.
    :param type_of_error: The type of error to simulate ('graph_overload' or 'node_malfunction').
    :return: A list of new datapoints with the simulated errors.
    """
    
    if type_of_error == "graph_overload":
        return ([graph_overload(datapoint, i/divide_by) for i in x_interval], [i/divide_by for i in x_interval])
    elif type_of_error == "node_malfunction":
        return ([node_malfunction(datapoint, i / len(datapoint.adjacency_matrix), 0) for i in x_interval], [i / len(datapoint.adjacency_matrix) for i in x_interval])

def graph_overload(datapoint, overload_factor):
    """
    Simulates graph overload by increasing the 'packages received' and 'packages sent' for a subset of nodes.

    :param datapoint: The data point representing the network.
    :param overload_factor: The factor by which to increase traffic (randomized for each node).
    :return: A new DataPoint with overloaded nodes.
    """
    
    new_features = np.copy(datapoint.features)
    num_nodes = len(new_features)
    mean = 10#num_nodes-5 # Mean is set to half the number of nodes
    std_dev = 2#3

    # Determine the number of nodes to update
    nodes_to_update_count = int(round(np.random.normal(mean, std_dev)))
    nodes_to_update_count = max(1, min(nodes_to_update_count, num_nodes))

    # Randomly select the nodes to update
    nodes_to_update = np.random.choice(num_nodes, nodes_to_update_count, replace=False)

    for index in nodes_to_update:
        node_features = new_features[index]
        increase_factor_received = np.random.uniform(1.0, 1.5) * overload_factor
        increase_factor_sent = np.random.uniform(1.0, 1.5) * overload_factor

        for i in range(0, len(node_features)-1, 2):
            node_features[i] += node_features[i] * increase_factor_received
            node_features[i+1] += node_features[i+1] * increase_factor_sent

    return DataPoint(new_features, np.copy(datapoint.adjacency_matrix))

def node_malfunction(datapoint, malfunction_rate, overload_factor):
    """
    Simulates node malfunction by specifically increasing 'packages received' and 'packages sent' 
    for a subset of nodes, using an overload factor to determine the increase.

    :param datapoint: The data point representing the network.
    :param malfunction_rate: The proportion of nodes that will malfunction.
    :param overload_factor: The factor by which to increase traffic for malfunctioning nodes.
    :return: A new DataPoint with malfunctioning nodes.
    """

    new_features = np.copy(datapoint.features)
    num_nodes = len(new_features)
    num_malfunction_nodes = int(num_nodes * malfunction_rate)
    mean = num_malfunction_nodes
    std_dev = 1

    # Determine the number of nodes to update
    nodes_to_update_count = int(round(np.random.normal(mean, std_dev)))
    nodes_to_update_count = max(1, min(nodes_to_update_count, num_nodes))

    # Randomly select the nodes to update
    nodes_to_update = np.random.choice(num_nodes, nodes_to_update_count, replace=False)

    for index in nodes_to_update:
        node_features = new_features[index]
        increase_factor_received = np.random.uniform(0, 2) * overload_factor
        increase_factor_sent = np.random.uniform(0, 2) * overload_factor

        for i in range(0, len(node_features)-1, 2):
            node_features[i] *= increase_factor_received
            node_features[i+1] *= increase_factor_sent

    return DataPoint(new_features, np.copy(datapoint.adjacency_matrix))