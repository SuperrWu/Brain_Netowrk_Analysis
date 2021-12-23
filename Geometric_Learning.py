from Graph_Representation import EEGDataset
import dgl
import torch
import dgl.nn.pytorch as dglnn
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        # 应用图卷积和激活函数
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['potentials'] = h
            # 使用平均读出计算图表示
            hg = dgl.mean_nodes(g, 'potentials')
            return self.classify(hg)





if __name__ == "__main__":
    dataset = EEGDataset()
    dataloader = GraphDataLoader(
        dataset,
        batch_size=16,
        drop_last=False,
        shuffle=True)
    model = Classifier(100, 20, 2)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(20):
        for batched_graph, labels in dataloader:
            feats = batched_graph.ndata['potentials']
            logits = model(batched_graph, feats)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
