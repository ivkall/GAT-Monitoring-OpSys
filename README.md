# GAT-Based Monitoring of Complex Operational Systems
Code implementation for Graph Attention Network-Based Monitoring of Complex Operational Systems.

## Installation
### Requirements
* Python >= 3.8.10
* Pytorch >= 1.13.0

### Install packages
```
pip install -r requirements.txt
```

## Usage
### Data preparation
The data should be csv file placed in /data with the following structure:
| No. | Time        | Source                    | Destination               | Protocol  | Length | Info                                      |
|-----|-------------|---------------------------|---------------------------|-----------|--------|-------------------------------------------|
| 1   | 0.000000000 | B&amp;RIndustria_4e:59:a1 | EPLv2_SoC                 | POWERLINK | 60     | 240-&gt;255 SoC                           |
| 2   | 0.000245055 | B&amp;RIndustria_4e:59:a1 | B&amp;RIndustria_33:55:02 | POWERLINK | 1504   | 240-&gt; 15 PReq [1480] F:RD=1,EA=0 V:0.0 |

Inside ```/extract_data/extract_graphv2.py``` or ```/extract_data/extract_graph_time.py``` replace ```file_name``` value with the the name of the csv file, and run the script.

The output will be a pkl file with the specified number of intervals.

### Run test case
Existing test cases can be found in ```generated_test_cases.ipynb``` and ```simulated_test_cases.ipynb```. Run the first cell with imports.

For the GAT model, ```one_graph_data``` is an attribute that contains the whole graph in one interval. The train and test data can have any number of intervals.

Example test case:
```
run_gat_model(one_graph_data="data1_1_big.pkl",
              train_interval_data="data1_100_time-1.0.pkl",
              test_interval_data="data1_100_time-1.0.pkl",
              gat_epochs=50,
              gat_batches=32,
              gat_lr=0.01,
              autoencoder_epochs=100,
              train=True,
              save_model=False,
              plot_features=True,
              test_individual=True)
```
