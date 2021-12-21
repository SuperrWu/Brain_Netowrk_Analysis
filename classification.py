#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import scipy.io as io
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from keras.models import Sequential
import tensorflow as tf



# In[2]:


path = "eegtrialsdata.mat"


# In[3]:


path


# In[4]:


data=io.loadmat(path)
train_x = data["trialsdata_1_3"][0]
train_y = data["trialsdata_1_3"][1]


# In[5]:


train_x[0].shape


# In[6]:


x_train = []
for i in train_x:
    x_train.append(i)
x_train = np.array(x_train)


# In[7]:


y_train = []
for i in train_y:
    if i[0][0] == 1:
        y_train.append(0)
    else:
        y_train.append(1)   
y_train = np.array(y_train)


# In[8]:


x_train = x_train[:,17:41, :]


# In[9]:


x_train


# In[10]:


def get_brain_connectivity_matrix(epochs):
    try:
        data = epochs.get_data(picks=["eeg"])
    except:
        data = epochs
    brain_network = []
    df = data
    for i in df:
        temp = []
        for j in df:
            temp.append(np.corrcoef(i, j)[0][1])
        brain_network.append(temp)
    brain_network = np.array(brain_network)
    threshold = np.quantile(brain_network, 0.7)
    for i in range(len(brain_network)):
        for j in range(len(brain_network)):
                if brain_network[i][j] < threshold:
                    brain_network[i][j] = 0
    for i in range(len(brain_network)):
        brain_network[i][i] = 0
    brain_network = normalize_data(brain_network)
    return brain_network


# In[11]:


def normalize_data(data):
        # print(data.shape)
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


# In[12]:


import networkx as nx
import mne 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
    
def cal_features(dictionary):
    df = pd.DataFrame(dictionary.values(), index = dictionary.keys())
    return df

def get_features_from_graph(data):
    G = nx.convert_matrix.from_numpy_matrix(brain_network)
    locs_info_path = 'C:\\Users\\25568\\Documents\\WeChat Files\\wxid_qkja1xdvzj2s22\\FileStorage\\File\\2021-10\\gtec_64_channels.loc'
    montage = mne.channels.read_custom_montage(locs_info_path)
    labels_dict =  dict(zip([i for i in range(len(montage.ch_names[17:41]))], montage.ch_names[17:41]))
    positions = montage.get_positions()["ch_pos"]
    labels_position_dict = {}
    for i in positions:
        if i in montage.ch_names[17:41]:
            labels_position_dict[i] = positions[i][:-1]
    G = nx.relabel_nodes(G, labels_dict)
    centrality = dict(zip(montage.ch_names[17:41], np.sum(brain_network, axis=1)))
    degree = dict(G.degree())
    btween_centrality = nx.betweenness_centrality(G,normalized=False)
    triangles = nx.triangles(G)
    closeness_centrality = nx.closeness_centrality(G)
    degree_df = cal_features(degree)
    centrality_df = cal_features(centrality)
    btween_centrality_df = cal_features(btween_centrality)
    triangles_df = cal_features(triangles)
    closeness_centrality_df = cal_features(closeness_centrality)
    df = pd.concat([degree_df, centrality_df, btween_centrality_df, triangles_df, closeness_centrality_df], axis=1)
    scaler=MinMaxScaler().fit(df)     #声明类，并用fit()方法计算后续标准化的mean与std
    X_scale=scaler.transform(df)
    nodes_size = dict(zip(montage.ch_names[17:41], df.mean(axis=1).to_list()))
    return G, nodes_size
    


# In[13]:


def get_train_data(brain_network):
    G, nodes_importance = get_features_from_graph(brain_network)
    row_data =[]
    for i in zip(brain_network, nodes_importance):
        rows = i[0]  * nodes_importance[i[1]]
        row_data.append(rows)
    row_data = np.array(row_data)
    new_brain_network = row_data.T
    columns_data = []
    for i in zip(new_brain_network, nodes_importance):
        columns = i[0]  * nodes_importance[i[1]]
        columns_data.append(rows)
    columns_data = np.array(columns_data).T
    columns_data = []
    for i in zip(new_brain_network, nodes_importance):
        columns = i[0]  * nodes_importance[i[1]]
        columns_data.append(rows)
    columns_data = np.array(columns_data).T
    return columns_data


# In[14]:


data=io.loadmat(path)
train_x = data["trialsdata_1_3"][0]
train_y = data["trialsdata_1_3"][1]


# In[15]:


X_train = []
for i in train_x:
    # print(train_x[i][17:41,:])
    current_data = i[17:41,:]
    brain_network = get_brain_connectivity_matrix(current_data)
    train_x = get_train_data(brain_network)
    train_x = train_x.flatten()
    X_train.append(train_x)


# In[16]:


X = np.array(X_train)


# In[17]:


Y = []
for i in train_y:
    if i[0][0] == 1:
        Y.append(0)
    else:
        Y.append(1)
Y = np.array(Y)


# In[18]:


X.shape


# In[19]:


Y.shape


# In[31]:


#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
#clf = RandomForestClassifier(max_depth=100, random_state=0)
#clf.fit(X_train, y_train)
#clf.score(X_test, y_test)


# In[20]:


#建立神经网络
model = Sequential()#先建立一个顺序模型
#向顺序模型里加入第一个隐藏层，第一层一定要有一个输入数据的大小，需要有input_shape参数
#model.add(Dense(n_hidden_1, activation='relu', input_shape=(n_input,)))
model.add(Dense(n_hidden_1, activation='relu', input_dim=n_input)) #这个input_dim和input_shape一样，就是少了括号和逗号
model.add(Dense(n_hidden_2, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])


# In[21]:


from keras.models import Sequential
import tensorflow as tf


# In[22]:


# In[ ]:




