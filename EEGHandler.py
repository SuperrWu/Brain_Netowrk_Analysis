import os
import scipy.io as io
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class EEGHandler(object):
    def __init__(self):
        pass

    def load_eeg(self, path):
        # load eeg data from mat file
        data=io.loadmat(path)
        train_x = data["trialsdata_1_3"][0]
        train_y = data["trialsdata_1_3"][1]
        x_train = []
        for i in train_x:
            x_train.append(i)
        x_train = np.array(x_train)
        y_train = []
        for i in train_y:
            if i[0][0] == 1:
                y_train.append(0)
            else:
                y_train.append(1)
        y_train = np.array(y_train)
        return x_train, y_train

    def compute_correlation_coefficient(self, i, j):
        # compute cc
        return np.corrcoef(i, j)[0][1]

    def compute_adjacency_matrix(self, data, threshold):
        # compute Adjacency Matrix
        adjacency_matrix = []
        for i in data:
            temp = []
            for j in data:
                temp.append(self.compute_correlation_coefficient(i, j))
            adjacency_matrix.append(temp)
        adjacency = np.array(adjacency_matrix)
        threshold_value = np.quantile(adjacency, threshold)
        
        # threshold
        for i in range(len(adjacency)):
            for j in range(len(adjacency)):
                if adjacency[i][j] < threshold_value:
                    adjacency[i][j] = 0
        # have no relationship with self
        for i in range(len(adjacency)):
            adjacency[i][i] = 0
        adjacency = self.normalize(adjacency)
        return adjacency

    def normalize(self, data):
        new_data = []
        for i in data:
            for j in i:
                if j != 0:
                    new_data.append(j)
        min = np.min(new_data)
        max = np.max(new_data)
        scaler = MinMaxScaler()
        train_x = scaler.fit_transform(np.array(new_data).astype(np.float32).reshape(-1, 1)).reshape(-1, len(new_data))
        count = 0
        for i in range(len(data)):
            for j in range(len(data)):
                if data[i][j] != 0:
                    data[i][j] = train_x[0][count]
                    count += 1
        return data
    


if __name__ == "__main__":
    dh = DataHandler()
    path = "eegtrialsdata.mat"
    x, y = dh.load_eeg(path)
    data = x[0]
    adjacency = dh.compute_adjacency_matrix(data, threshold = 0.7)
    print(adjacency)


    
