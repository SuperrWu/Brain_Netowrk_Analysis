from EEGHandler import EEGHandler
import networkx as nx
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from CreateGraph import BrainNetwork
import dgl
from dgl.data import DGLDataset
from tqdm import tqdm
import torch

class EEGDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='EEGDataset')

    def process(self):
        self.graphs = []
        self.labels = []
        EEGhandler = EEGHandler()
        path = "eegtrialsdata.mat"
        x, y = EEGhandler.load_eeg(path)
        for i in tqdm(zip(x, y)):
            # print(idx, item)
            functional_connectivity = EEGhandler.compute_functional_connectivity(i[0])
            adjacency = EEGhandler.thresholding(functional_connectivity, 0.6, ignore_negative = False)
            brainnet = BrainNetwork(adjacency, i[0], nodes_features=True)
            g = dgl.from_networkx(brainnet.Graph)
            g.ndata['potentials'] = torch.from_numpy(i[0])
            # g.edata["weight"] = adjacency
            # for idx, item in g.nodes():
            #    g.ndata["potentials"] = torch.ones(g.num_nodes(), 3)
            # print(g.ndata["potentials"][0])
            self.graphs.append(g)
            self.labels.append(i[1])


        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    @property
    def num_labels(self):
        return 2

    @property
    def num_graphs(self):
        return len(self.graphs)


if __name__ == "__main__":
    dataset = EEGDataset()
    
    # print(graph.ndata["potentials"][1].shape)
