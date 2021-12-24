# About

Analyzing the brain relationship and classifying directions by using Brain Connectivity Network(or functional connectivity matrix).

# How?
##  Construct BrainConnectivityNet

* Compute Pearson Correlation Coefficient of paired channel to build functional connectivity matrix.

* Thresholding (considering negative parts or not) to Adjacency matrix

* Convert adjacency matrix to NetworkX graph object

## Analysis

* MATLAB

## Classification

- Common way

(1) Average every trial and every channel's potentials in all time(1s)

(2) Construct one BrainConnectivityNet

(3) Extract edges and nodes features(centrality, degree, etc.)

(4) Extract nodes features by using PCA

(5) Integrate edges and nodes features [edges * in-nodes:importance of nodes * out-nodes:importance of nodes]

(6) Flatten features (62 * 62 = 3844)

(7) Train a forward nerual network
- Improved way

(1) Extract time features by building several BrainConnectivityNet by dividing time into pieces (10 time point out of 100 time point, 100HZ here)

(2) Do same thing before to get trianing data with format (60 sample, 10 time step, 3844 features)

(3) Train a LSTM nerual network
-  GNN

(1) Convert networkx object to DGL data.Graph object

(2) Apply GNN

# Usage

Install necessary packages

```
pip install -r requirement.txt
```
```
EEGhandler = EEGHandler()
# your eeg raw data
path = "eegtrialsdata_hyh.mat"
# load data
x, y = EEGhandler.load_eeg(path)

for i in tqdm(x):
  # compute functional_connectivity
  functional_connectivity = EEGhandler.compute_functional_connectivity(i)
  # compute adjacency by thresholding, considering negative parts
  adjacency = EEGhandler.thresholding(functional_connectivity, 0.6, ignore_negative = False)
  # construct brainnet
  BrainNet = BrainNetwork(adjacency)
  # features extructor
  feats = FeatureExtractor()
  nodes_feats = feats.get_nodes_features(G)
  edges_feats = feats.get_edges_features(G)
  features = feats.integrate_features(nodes_feats, edges_feats)
  # normalization
  scaler = MinMaxScaler().fit(features)
  X_scale = scaler.transform(features)
```


