import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import copy
import matplotlib.pyplot as plt
import tensorflow as tf

from utilities.datapoint import DataPoint
from custom_gat.custom_GAT import GAT, train, extract_all_features, test_with_autoencoder, train_gat_on_multiple_graphs
from utilities.autoencode import Autoencoder
import utilities.statistic as stat

def run_gat_model(one_graph_data="standard_1_big.pkl", train_interval_data="data1_100.pkl", test_interval_data="data1_100.pkl", test_labels=None, test_iterations=5, type_of_error=None, gat_epochs=50, gat_batches=32, gat_lr=0.01, autoencoder_epochs=100, train=False, save_model=False, plot_features=False, test_individual=False, exclusion_percentile=10):

    training_data_one_graph=stat.load_data(f"data/{one_graph_data}")
    training_data = stat.load_data(f"data/{train_interval_data}")
    testing_data = stat.load_data(f"data/{test_interval_data}")
    
    #__________________________________________________________________________________________________________________
    # GAT
    #__________________________________________________________________________________________________________________
    import matplotlib.pyplot as plt
    import re
    
    match = re.search(r'(?<=-)\d+\.\d+', test_interval_data)
    if match:
        interval_length = float(match.group())
        X = [i * interval_length for i in range(1,1+len(testing_data))]
        
    if plot_features:
        # PLOT_DATA = [np.sum(np.sum(interval.features)) for interval in training_data]
        # plt.figure(figsize=(10, 2))
        # plt.plot(X, PLOT_DATA)
        # plt.title("Training data, Sum of features")
        # plt.show()

        PLOT_DATA = [np.sum(np.sum(interval.features)) for interval in testing_data]
        plt.figure(figsize=(10, 2))
        plt.plot(X, PLOT_DATA)
        plt.title("Testing data, Sum of features")
        plt.show()
   
    nbr_of_features_per_node = np.array(training_data_one_graph[0].features).shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gat_model = GAT(in_dim=nbr_of_features_per_node,
                    hidden_dim=64,
                    out_dim=nbr_of_features_per_node,
                    num_heads=8)

    if train:
        gat_model = train_gat_on_multiple_graphs(device, gat_model, training_data, gat_epochs, learning_rate=gat_lr, batch_size=gat_batches)
    else:
        gat_model = stat.load_data("custom_gat/saved_model/gat.pkl")
    #__________________________________________________________________________________________________________________
    # SPLIT TRAINING DATA
    #__________________________________________________________________________________________________________________

    test_data_point = DataPoint(copy.deepcopy( training_data[20].features ) , copy.deepcopy( training_data[20].adjacency_matrix) )
    all_aggregated_features = extract_all_features(gat_model, training_data)
    
    train_mean = np.mean(all_aggregated_features.detach().numpy(), axis=0)
    train_std = np.std(all_aggregated_features.detach().numpy(), axis=0)
    all_aggregated_features_scaled = stat.standardize_data (all_aggregated_features.detach().numpy())
    train_data, val_data = train_test_split(all_aggregated_features_scaled, test_size=0.2)

    #__________________________________________________________________________________________________________________
    # AUTOENCODER
    #__________________________________________________________________________________________________________________

    if train:
        autoencoder = Autoencoder(input_dim=nbr_of_features_per_node,
                                  encoding_dim=32)
        train_data = torch.tensor(all_aggregated_features_scaled, dtype=torch.float32).detach().numpy()
        val_data = torch.tensor(val_data, dtype=torch.float32).detach().numpy()
        history = autoencoder.train_and_evaluate(train_data, val_data, epochs=autoencoder_epochs)
    else:
        loaded_autoencoder = tf.keras.models.load_model("custom_gat/saved_model/ae.keras")
        autoencoder = Autoencoder(input_dim=nbr_of_features_per_node,
                                  encoding_dim=32,
                                  loaded_autoencoder=loaded_autoencoder)
    
    if save_model:
        stat.save_data("custom_gat/saved_model/gat.pkl", gat_model)
        autoencoder.autoencoder.save("custom_gat/saved_model/ae.keras")
    #__________________________________________________________________________________________________________________
    # TEST MODEL ON GENERATED ERRORS
    #__________________________________________________________________________________________________________________

    if type_of_error:
        if type_of_error == "graph_overload":
            x_interval = range(0,10)
        else:
            x_interval = range(0,48,4)
        
        result_max = []
        result_mse = []
        result_x = []
        import utilities.generateTestData as gtd
        from collections import defaultdict
        col_dict = defaultdict(list)
        divide_by = 10
        
        for _ in range(test_iterations):
            datapoints, factors = gtd.generateData(test_data_point, 0, type_of_error, divide_by, x_interval)
            x = [i/divide_by if type_of_error == "graph_overload" else i/48 for i in x_interval]
            for j, datapoint in enumerate(datapoints):
                prediction, max_error, mse_error = test_with_autoencoder(gat_model, train_mean, train_std, autoencoder, datapoint)
    
                result_max.append(max_error)
                result_mse.append(mse_error)
                col_dict[j].append(mse_error)
            result_x.append(x)
        
        result_x_flat = np.array(result_x).flatten()
        result_max = np.array(result_max)
        result_mse = np.array(result_mse)

        correlation_matrix = np.corrcoef(result_x_flat, result_mse)
        correlation_coefficient = correlation_matrix[0, 1]
        print("Correlation Coefficient:", correlation_coefficient)
        
        threshold = 1.05*np.mean(col_dict[0])
        print(threshold)
        ratios = [(col > threshold).sum()/test_iterations for col in col_dict.values()]
        print(f"{type_of_error} factors: {' & '.join([f'{factor:.3f}' for factor in factors])}")
        print(f"ratio above threshold: {' & '.join([f'{r:.2f}' for r in ratios])} & {threshold:.4f} & {correlation_coefficient:.4f} \n")



        
        import matplotlib.pyplot as plt
    
        plt.scatter(result_x_flat, result_mse, s=1)
        plt.axhline(y=threshold, color='r', linestyle='--')
        # plt.yscale('log')
        plt.title(f"GAT Simulated error: {type_of_error}")
        plt.xlabel("Datapoint")
        plt.ylabel("MSE Error")

    #__________________________________________________________________________________________________________________
    # TEST MODEL ON INDIVIDUAL DATAPOINTS
    #__________________________________________________________________________________________________________________
    
    if test_individual:
        losses = []
        for datapoint in testing_data:
            prediction, max_error, mse_error = test_with_autoencoder(gat_model, train_mean, train_std, autoencoder, datapoint)
            losses.append(mse_error)
        
        import matplotlib.pyplot as plt
    
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

        gaussian_blur=False
        if gaussian_blur:
            from scipy.ndimage import gaussian_filter1d
            plt.plot(X, gaussian_filter1d(losses, sigma=1), linewidth=1)
        
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
        plt.title(f"Classified anomalies")
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