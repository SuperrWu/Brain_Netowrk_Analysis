import os
import pickle 
import numpy as np 
from CreateGraph import BrainNetwork 
import matplotlib.pyplot as plt
import seaborn as sns
from GetTrainData import FeatureExtractor
import networkx as nx



class BrainViwer(object):
    def __init__(self):
        self.data = self.get_data()
        self.BrainNet = BrainNetwork(self.data)

    def get_data(self):
        with open("train_x.pkl", "rb") as f:
            data = pickle.load(f)
        f.close()
        data = np.mean(data, axis = 0)
        return data

    def plot_heatmap(self):
        sns.heatmap(self.BrainNet.adjacency, xticklabels=self.BrainNet.channel_dict.values(), yticklabels=self.BrainNet.channel_dict.values())
        plt.savefig("heatmap.PNG")

    def plot_networks(self):
        feats = FeatureExtractor()
        # dictionary = feats.get_nodes_features(self.BrainNet)
        dictionary = self.BrainNet.get_centrality()
        de2 = [dictionary[v]*30 for v in sorted(dictionary.keys(), reverse=False)] 
        nx.draw_networkx(self.BrainNet.Graph, self.BrainNet.channel_position_dict, node_size=de2, width=[float(v['weight']) for (r, c, v) in self.BrainNet.Graph.edges(data=True)])
        plt.savefig("brainnetworks.PNG")





if __name__ == "__main__":
    viwer = BrainViwer()
    # viwer.plot_heatmap()i
    viwer.plot_networks()



