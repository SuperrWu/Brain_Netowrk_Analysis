from EEGHandler import EEGHandler
import networkx as nx
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from CreateGraph import BrainNetwork
import dgl



if __name__ == "__main__":
    EEGhandler = EEGHandler()
    path = "eegtrialsdata.mat"
    x, y = EEGhandler.load_eeg(path)
    data = x[0]
    functional_connectivity = EEGhandler.compute_functional_connectivity(data)
    adjacency = EEGhandler.thresholding(functional_connectivity, 0.6, ignore_negative = False)
    brainnet = BrainNetwork(adjacency)
    dgl.from_networkx(brainnet.Graph)
