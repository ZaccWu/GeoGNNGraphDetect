import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class MultiRelationalGNN(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, aggr="mean"):
        super(MultiRelationalGNN, self).__init__(aggr=aggr)
        self.num_relations = num_relations
        # different edge type have different weights
        self.rel_weight = nn.ModuleList(
            [nn.Linear(hidden_channels, out_channels, bias=False) for _ in range(num_relations)]
        )
        # project all the node features into the same latent space
        self.field_map = nn.Linear(in_channels, hidden_channels, bias=True)

    def forward(self, x, edge_index, edge_attr):
        x_trans = self.field_map(x)
        out = self.propagate(edge_index, x=x_trans, edge_attr=edge_attr)
        return out

    def message(self, x_j, edge_attr):
        edge_type = edge_attr.squeeze().long()  # edge attr to int: [num_edges]
        out = torch.zeros(len(x_j), 5) ## TODO: modified dimension settings
        for rel in range(self.num_relations):
            mask = edge_type == rel  # edge_type 'rel'
            if mask.any():
                out[mask] = self.rel_weight[rel](x_j[mask])  # 应用 rel 对应的权重矩阵
        return out

    def update(self, aggr_out): # neighbors' information after aggregation
        return aggr_out