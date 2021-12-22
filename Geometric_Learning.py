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

class SyntheticDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='synthetic')

    def process(self):
        self.graphs = []
        self.labels = []
        EEGhandler = EEGHandler()
        path = "eegtrialsdata.mat"
        x, y = EEGhandler.load_eeg(path)
        for i in tqdm(x):
            functional_connectivity = EEGhandler.compute_functional_connectivity(i)
            adjacency = EEGhandler.thresholding(functional_connectivity, 0.6, ignore_negative = False)
            brainnet = BrainNetwork(adjacency)
            g = dgl.from_networkx(brainnet.Graph)
            self.graphs.append(g)
            self.labels.append(0)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


if __name__ == "__main__":
    dataset = SyntheticDataset()
    graph, label = dataset[0]
    print(graph, label)

