import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
from utilities.datapoint import DataPoint
from utilities.autoencode import Autoencoder
from custom_gcn.custom_GCN import GCN, DataPoint, extract_features, test_with_autoencoder
import utilities.statistic as stat

def run_gcn_model(train_interval_data="data1_100.pkl", test_interval_data="data1_100.pkl", test_labels=None, test_iterations=5, type_of_error=None, x_interval=range(0,10), autoencoder_epochs=200, train=False, save_model=False, test_individual=False, exclusion_percentile=10):
    
    training_data = stat.load_data(f"data/{train_interval_data}")
    testing_data = stat.load_data(f"data/{test_interval_data}")
    
    #__________________________________________________________________________________________________________________
    # GCN
    #__________________________________________________________________________________________________________________
    
    test_data_point = DataPoint(copy.deepcopy( training_data[20].features ) , copy.deepcopy( training_data[20].adjacency_matrix) )

    nbr_of_nodes = np.array(training_data[0].features).shape[0]
    nbr_of_features_per_node = np.array(training_data[0].features).shape[1]
    print(nbr_of_features_per_node)
    output_dim = 12

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gcn_model = GCN(input_dim=nbr_of_features_per_node,
                    hidden_dim1=8*nbr_of_features_per_node,
                    hidden_dim2=4*nbr_of_features_per_node,
                    output_dim=3*nbr_of_features_per_node).to(device)
    gcn_model.eval()

    gcn_transformed_features = extract_features(device, gcn_model, training_data)
    train_mean = np.mean(gcn_transformed_features, axis=0)
    train_std = np.std(gcn_transformed_features, axis=0)
    gcn_transformed_features_scaled = stat.standardize_data(gcn_transformed_features)

    #__________________________________________________________________________________________________________________
    # AUTOENCODER
    #__________________________________________________________________________________________________________________
    
    nmbr_of_features_per_node = gcn_transformed_features.shape[1]
    if train:
        train_data, val_data = train_test_split(gcn_transformed_features_scaled, test_size=0.2)
        autoencoder = Autoencoder(input_dim=nmbr_of_features_per_node,
                                  encoding_dim=3*nbr_of_features_per_node)
        history = autoencoder.train_and_evaluate(train_data, val_data, epochs=autoencoder_epochs)
    else:
        loaded_autoencoder = tf.keras.models.load_model("custom_gcn/saved_model/ae.keras")
        autoencoder = Autoencoder(input_dim=nmbr_of_features_per_node,
                                  encoding_dim=3*nbr_of_features_per_node,
                                  loaded_autoencoder=loaded_autoencoder)
    
    if save_model:
        autoencoder.autoencoder.save("custom_gcn/saved_model/ae.keras")
    #__________________________________________________________________________________________________________________
    # TEST MODEL ON GENERATED ERRORS
    #__________________________________________________________________________________________________________________

    if type_of_error:
        if type_of_error == "graph_overload":
            x_interval = range(0,3)
        else:
            x_interval = range(0,48,4)
            
        result_max = []
        result_mse = []
        result_x = []
        import utilities.generateTestData as gtd
        from collections import defaultdict
        col_dict = defaultdict(list)
        divide_by = 10
        
        for i in range(test_iterations):
            datapoints, factors = gtd.generateData(test_data_point, 0, type_of_error, divide_by, x_interval)
            x = [i/divide_by for i in x_interval]
            for j, datapoint in enumerate(datapoints):
                test_data_gcn_features = extract_features(device, gcn_model, [datapoint])
                
                test_data_gcn_features_scaled = stat.standardize_data(test_data_gcn_features, train_mean, train_std)
                test_data_prediction, _, _ = autoencoder.evaluate(test_data_gcn_features_scaled)
                max_error = np.max(test_data_gcn_features_scaled - test_data_prediction)
                mse_error = mean_squared_error(test_data_gcn_features_scaled, test_data_prediction)
                result_max.append(max_error)
                result_mse.append(mse_error)
                col_dict[j].append(mse_error)
            result_x.append(x)
        
        result_x = np.array(result_x).flatten()
        result_max = np.array(result_max)
        result_mse = np.array(result_mse)

        correlation_matrix = np.corrcoef(result_x, result_mse)
        correlation_coefficient = correlation_matrix[0, 1]
        print("Correlation Coefficient:", correlation_coefficient)
        
        threshold = 1.05*np.mean(col_dict[0])
        print(threshold)
        ratios = [(col > threshold).sum()/test_iterations for col in col_dict.values()]
        print(f"{type_of_error} factors: {' & '.join([f'{factor:.3f}' for factor in factors])}")
        print(f"ratio above threshold: {' & '.join([f'{r:.2f}' for r in ratios])} & {threshold:.4f} & {correlation_coefficient:.4f} \n")
        
        import matplotlib.pyplot as plt
        plt.scatter(result_x, result_mse, s=1)
        plt.axhline(y=threshold, color='r', linestyle='--')
        # plt.yscale('log')
        plt.title("GCN Simulated error")
        plt.xlabel(type_of_error)
        plt.ylabel("MSE Error")

    #__________________________________________________________________________________________________________________
    # TEST MODEL ON INDIVIDUAL DATAPOINTS
    #__________________________________________________________________________________________________________________
    
    if test_individual:
        losses = []
        for datapoint in testing_data:
            prediction, max_error, mse_error = test_with_autoencoder(device, gcn_model, [datapoint], train_mean, train_std, autoencoder)
            losses.append(mse_error)
        
        import matplotlib.pyplot as plt
        import re
    
        match = re.search(r'(?<=-)\d+\.\d+', test_interval_data)
        if match:
            interval_length = float(match.group())
            X = [i * interval_length for i in range(1,1+len(testing_data))]
    
        plt.figure(figsize=(10, 2))
        plt.plot(X, losses, linewidth=1)
        plt.yscale('log')
        plt.title(f"Individual datapoint loss")
        plt.xlabel("Time (s)")
        plt.ylabel("MSE Error")
        
        if test_labels:
            with open(f"data/{test_labels}", "r") as file:
                start_time = float(file.readline().strip())
                end_time = float(file.readline().strip())
            print(start_time, end_time)
            plt.axvspan(start_time, end_time, color='red', alpha=0.1)
    
        losses_non_anomalies = []
        for i, t in enumerate(X):
            if t < start_time or t > end_time:
                losses_non_anomalies.append(losses[i])
        # Calculate mean and standard deviation of the data
        analyzer = stat.DataStreamAnalyzer(exclusion_percentile)
        analyzer.add_data(losses_non_anomalies)
        
        mean, std_dev = analyzer.calculate_stats()
        if mean is not None and std_dev is not None:
            print("Mean:", mean, "Std:", std_dev)
        else:
            print("Insufficient data after filtering.")
            
        # Calculate mean and standard deviation of the data
        # mean = np.mean(losses)
        # std_dev = np.std(losses)
    
        # Define the thresholds based on different standard deviations from the mean
        stds = [1, 2, 10]
        thresholds = [mean + k*std_dev for k in stds]
    
    
    
        # Add vertical lines for each threshold
        colors = ['green', 'orange', 'red']  # Adjust colors as needed
        linestyles = ['--', '--', '--']  # Adjust linestyles as needed
        for k, threshold, color, linestyle in zip(stds, thresholds, colors, linestyles):
            plt.axhline(y=threshold, color=color, linestyle=linestyle, label=f'{k} std dev', linewidth=1)
    
        plt.legend()
        plt.show()
    
        # Plot anomalies
        anomalies1 = np.zeros_like(losses)
        anomalies2 = np.zeros_like(losses)
        anomalies3 = np.zeros_like(losses)
        anomalies1[losses > thresholds[0]] = 1
        anomalies2[losses > thresholds[1]] = 2
        anomalies3[losses > thresholds[2]] = 3
    
        plt.figure(figsize=(10, 1))
        plt.plot(X, anomalies1, linewidth=1, label="MSE > 1 std dev", color='green')
        plt.plot(X, anomalies2, linewidth=1, label="MSE > 2 std dev", color='orange')
        plt.plot(X, anomalies3, linewidth=1, label="MSE > 10 std dev", color='red')
        plt.title(f"Classified anomalies (MSE > {k} std)")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        if test_labels:
            with open(f"data/{test_labels}", "r") as file:
                start_time = float(file.readline().strip())
                end_time = float(file.readline().strip())
            print(start_time, end_time)
            plt.axvspan(start_time, end_time, color='red', alpha=0.1)

            for ind, anoms in enumerate([anomalies1, anomalies2, anomalies3]):
                TP = 0
                FP = 0
                TN = 0
                FN = 0
                for i, a in enumerate(anoms):
                    if a == ind + 1:
                        if start_time <= X[i] <= end_time:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        if start_time <= X[i] <= end_time:
                            FN += 1
                        else:
                            TN += 1
                f1 = 2*TP/(2*TP+FP+FN)
                print(f"f1-score {ind+1}: {f1}")
        plt.legend()
        plt.show()