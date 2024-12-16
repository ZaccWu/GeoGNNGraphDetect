import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, h_dim=8):
        super().__init__()
        self.dropout = 0.5
        self.gatconv_1 = GATConv(in_dim, h_dim, dropout=self.dropout)
        self.gatconv_2 = GATConv(h_dim, h_dim, dropout=self.dropout)
        self.linear = nn.Linear(h_dim, 1)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_type, edge_time):
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = self.gatconv_1(x, edge_index)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.gatconv_2(x1, edge_index)
        out = self.act(self.linear(x2))  # x3: (batch*num_stock, hidden)
        return out



