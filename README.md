# About

Analyzing the brain relationship and classifying directions by using Brain Connectivity Network(or functional connectivity matrix).

# How?
##  construct BrainConnectivityNet

* Compute Pearson Correlation Coefficient of paired channel to build functional connectivity matrix.

* Thresholding (considering negative parts or not) to Adjacency matrix

* Convert adjacency matrix to NetworkX graph object

## Analysis

* MATLAB

## Classification

- Common way

* Average every trial and every channel's potentials in all time(1s)

* Construct one BrainConnectivityNet

* Extract edges and nodes features(centrality, degree, etc.)

* Extract nodes features by using PCA

* Integrate edges and nodes features [edges * in-nodes:importance of nodes * out-nodes:importance of nodes]

* Flatten features (62 * 62 = 3844)

* Train a forward nerual network

- Improved way

* Extract time features by building several BrainConnectivityNet by dividing time into pieces (10 time point out of 100 time point, 100HZ here)

* Do same thing before to get trianing data with format (60 sample, 10 time step, 3844 features)

* Train a LSTM nerual network

- GNN

* Convert networkx object to DGL data.Graph object

* Apply GNN

# Usage

install necessary packages

```
pip install -r requirement.txt
```


