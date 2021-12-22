from EEGHandler import EEGHandler
import networkx as nx
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class BrainNetwork(object):
    def __init__(self, adjacency = None):
        self.adjacency = adjacency
        self.Graph = self.convert_from_matrix(adjacency)
        # print(adjacency.shape)
        self.channel_dict, self.channel_position_dict = self.get_channel_locations("gtec_64_channels.loc")
    
    def convert_from_matrix(self, adjacency):
        G = nx.convert_matrix.from_numpy_matrix(adjacency)
        return G

    def get_channel_locations(self, file_path):
        montage = mne.channels.read_custom_montage(file_path)
        labels_dict =  dict(zip([i for i in range(len(montage.ch_names[:-2]))], montage.ch_names[:-2]))
        # print(labels_dict)
        positions = montage.get_positions()["ch_pos"]
        
        labels_position_dict = {}
        for i in positions:
            if i in montage.ch_names[:-2]:
                labels_position_dict[i] = positions[i][:-1]
        self.Graph = nx.relabel_nodes(self.Graph, labels_dict)
        # print("postions_dict:", labels_position_dict)
        return labels_dict, labels_position_dict
        

    def get_centrality(self):
        return dict(zip(list(self.channel_dict.keys()), np.sum(self.adjacency, axis=1)))

    def get_degree(self):
        return dict(self.Graph.degree())

    def get_between_centrality(self):
        return nx.betweenness_centrality(self.Graph,normalized=False)

    def get_triangles(self):
        return nx.triangles(self.Graph)

    def get_closeness_centrality(self):
        return nx.closeness_centrality(self.Graph)
    
    def plot_heatmap(self, path):
        sns.heatmap(self.adjacency, xticklabels=self.channel_dict.values() ,yticklabels=self.channel_dict.values())
        plt.savefig(path)

if __name__ == "__main__":
    EEGhandler = EEGHandler()
    path = "eegtrialsdata.mat"
    x, y = EEGhandler.load_eeg(path)
    data = x[0]
    adjacency = EEGhandler.compute_adjacency_matrix(data, threshold = 0.7)
    G = BrainNetwork(adjacency)
    print(G.get_centrality(G))

