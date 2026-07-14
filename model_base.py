import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv, GINConv, RGATConv, EGConv


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, h_dim=64):
        super().__init__()
        self.dropout = 0.5
        self.gatconv_1 = GATConv(in_dim, h_dim, dropout=self.dropout)
        self.gatconv_2 = GATConv(h_dim, h_dim, dropout=self.dropout)
        self.linear = nn.Linear(h_dim, 1)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_type):
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = self.gatconv_1(x, edge_index)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.gatconv_2(x1, edge_index)
        out = self.act(self.linear(x2))  # x3: (batch*num_stock, hidden)
        return out

class GIN(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, h_dim=64):
        super().__init__()
        self.ginconv_1 = GINConv(nn.Linear(in_dim, h_dim))
        self.ginconv_2 = GINConv(nn.Linear(h_dim, h_dim))
        self.linear = nn.Linear(h_dim, 1)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_type):
        x1 = self.ginconv_1(x, edge_index)
        x2 = self.ginconv_2(x1, edge_index)
        out = self.act(self.linear(x2))  # x3: (batch*num_stock, hidden)
        return out

class RGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, h_dim=32):
        super().__init__()
        self.rgatconv_1 = RGATConv(in_dim, h_dim, num_relations)
        self.rgatconv_2 = RGATConv(h_dim, h_dim, num_relations)
        self.linear = nn.Linear(h_dim, 1)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_type):
        x1 = self.rgatconv_1(x, edge_index, edge_type)
        x2 = self.rgatconv_2(x1, edge_index, edge_type)
        out = self.act(self.linear(x2))  # x3: (batch*num_stock, hidden)
        return out

class EGC(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, h_dim=64):
        super().__init__()
        self.egconv_1 = EGConv(in_dim, h_dim)
        self.egconv_2 = EGConv(h_dim, h_dim)
        self.linear = nn.Linear(h_dim, 1)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_type):
        x1 = self.egconv_1(x, edge_index)
        x2 = self.egconv_2(x1, edge_index)
        out = self.act(self.linear(x2))  # x3: (batch*num_stock, hidden)
        return out