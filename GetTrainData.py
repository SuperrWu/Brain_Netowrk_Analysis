from EEGHandler import EEGHandler
from CreateGraph import BrainNetwork
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn
from tqdm import tqdm
import numpy as np
import pickle

class FeatureExtractor(object):
    def __init__(self):
        pass

    # extract nodes features from a graph
    def get_nodes_features(self, Graph):
        centrality = self.convert_dict_to_df(Graph.get_centrality())
        degree = self.convert_dict_to_df(Graph.get_degree())
        between_centrality = self.convert_dict_to_df(Graph.get_between_centrality())
        triangles = self.convert_dict_to_df(Graph.get_triangles())
        closeness_centrality = self.convert_dict_to_df(Graph.get_closeness_centrality())
        df = pd.concat([ centrality ,  degree , between_centrality , triangles, closeness_centrality], axis=1)
        scaler = MinMaxScaler().fit(df)
        X_scale = scaler.transform(df)
        nodes_features = dict(zip(list(Graph.channel_dict.values()), df.mean(axis=1).to_list()))
        return nodes_features

    
    def convert_dict_to_df(self, dictionary):
        return pd.DataFrame(dictionary.values())


    # extract edges features from a graph
    def get_edges_features(self, Graph):
        return Graph.adjacency

    #  feature integration
    def integrate_features(self, nodes_feats, edges_feats):
        # print(edges_feats)
        for i in range(edges_feats.shape[0]):
            for j in range(edges_feats.shape[1]):
                edges_feats[i][j] = edges_feats[i][j] * float(list(nodes_feats.values())[i]) * float(list(nodes_feats.values())[j])
        scaler = MinMaxScaler().fit(edges_feats)
        X_scale = scaler.transform(edges_feats)
        return X_scale


if __name__ == "__main__":
    EEGhandler = EEGHandler()
    path = "eegtrialsdata.mat"
    x, y = EEGhandler.load_eeg(path)
    X = []
    for i in tqdm(x):
        adjacency = EEGhandler.compute_adjacency_matrix(i, threshold = 0)
        G = BrainNetwork(adjacency)
        feats = FeatureExtractor()
        nodes_feats = feats.get_nodes_features(G)
        edges_feats = feats.get_edges_features(G)
        features = feats.integrate_features(nodes_feats, edges_feats)
        X.append(features)
    X = np.array(X)
    with open("train_x.pkl", "wb") as f:
        pickle.dump(X, f)
    f.close()
    with open("train_y.pkl", "wb") as f:
        pickle.dump(y, f)
    f.close()
    


