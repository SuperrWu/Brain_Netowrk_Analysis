from Graph_Representation import EEGDataset
import dgl
import torch
import dgl.nn.pytorch as dglnn
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, in_feat, our_feat, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feat, out_feat)
        self.conv2 = dglnn.GraphConv(out_feat, out_feat)
        self.classify = nn.Linear(out_feat, lasses)

    def forward(self, g, h):
        # 应用图卷积和激活函数
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['potentials'] = h
            # 使用平均读出计算图表示
            hg = dgl.mean_nodes(g, 'potentials')
            hg= hg.squeeze(-1)
            return self.classify(hg)





if __name__ == "__main__":
    # Getting dataset
    dataset = EEGDataset()
    input_shape = dataset[0][0].ndata['potentials'].shape[1]
    n_classes = dataset.num_labels
    
    # define batched dataset
    dataloader = GraphDataLoader(
        dataset,
        batch_size=16,
        drop_last=False,
        shuffle=True)
    # create model
    model = Classifier(input_shape, 128, n_classes-1)
    opt = torch.optim.Adam(model.parameters())
    epoch_num = 150

    for epoch in range(epoch_num):
        # print(epoch)
        # epochs _losses = []
        for batched_graph, labels in dataloader:
            feats = batched_graph.ndata['potentials']
            pred = model(batched_graph, feats)
            loss = F.binary_cross_entropy(pred, labels)
            opt.zero_grad()
            loss.backward()
            # print(epoch, loss.detach().item())
            opt.step()
        print("Epoch ", epoch, "/150")
        print("-Loss: ", loss.detach().item())
