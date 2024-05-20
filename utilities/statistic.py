import numpy as np
import pickle

def standardize_data(data, mean=None, std=None):
    """
    Standardizes the data to have a mean of 0 and standard deviation of 1.
    If mean and std are provided, uses them to standardize the data.
    
    Parameters:
    data (np.array): The data to be standardized.
    mean (np.array): Optional. The mean to use for standardization.
    std (np.array): Optional. The standard deviation to use for standardization.

    Returns:
    np.array: The standardized data.
    """
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
    
    std[std == 0] = 1  # To avoid division by zero for constant features
    standardized_data = (data - mean) / std
    return standardized_data

def load_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def save_data(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

class DataStreamAnalyzer:
    def __init__(self, exclusion_percentile=10):
        """
        Initialize the DataStreamAnalyzer.
        
        :param exclusion_percentile: Percentage of extremal data points to exclude from each end.
        """
        self.data = []
        self.exclusion_percentile = exclusion_percentile

    def add_data(self, new_data):
        """
        Add a new array of values to the data stream.
        
        :param new_data: An iterable of new data points to add.
        """
        self.data.extend(new_data)
    
    def calculate_stats(self):
        """
        Calculate the mean and standard deviation of the data, excluding the specified percentile of extremal values.
        
        :return: A tuple containing the mean and standard deviation of the data excluding extremal values, or None if insufficient data.
        """
        if len(self.data) == 0:
            return None  # Return None if no data has been added
        
        lower_bound = np.percentile(self.data, self.exclusion_percentile)
        upper_bound = np.percentile(self.data, 100 - self.exclusion_percentile)
        filtered_data = [x for x in self.data if lower_bound <= x <= upper_bound]
        
        if not filtered_data:  # Check if all data are filtered out
            return None
        
        mean_val = np.mean(filtered_data)
        std_val = np.std(filtered_data)
        return mean_val, std_val