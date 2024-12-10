import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class MultiRelationGNN(MessagePassing):
    def __init__(self, in_dim, out_dim, num_relations, h_dim=16,
                 aggr='add', lambda_sym=0.5, alpha=0.1, beta=0.1):
        super(MultiRelationGNN, self).__init__(aggr=aggr)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations  # Number of relation types
        self.lambda_sym = lambda_sym  # Weight for symmetric part
        self.alpha = alpha  # Spatial decay parameter
        self.beta = beta  # Temporal decay parameter

        self.field_mlp = nn.Linear(in_dim, h_dim)

        # Separate MLPs for each relation type
        self.relation_mlps = torch.nn.ModuleList(
            [nn.Linear(h_dim * 2, h_dim) for _ in range(num_relations)]
        )
        self.mlp = nn.Linear(h_dim, 5) # target num=5

    def forward(self, x, edge_index, edge_type, edge_time, pos):
        # x (num_nodes, node_feature_dim): Node features
        # edge_index (2, num_edges):  Edge indices
        # edge_type (num_edges, 1):   Edge type information
        # edge_time (num_edges):      Edge timestamps
        # pos (num_nodes, pos_feature_dim): Node spatial positions

        row, col = edge_index  # row: source nodes, col: target nodes
        edge_time = edge_time.view(-1,1)
        node_emb0 = self.field_mlp(x)

        return self.propagate(edge_index=edge_index, x=node_emb0,
                              pos_k=pos[row], pos_l=pos[col], edge_type=edge_type, edge_time=edge_time)

    def message(self, x_j, x_i, pos_k, pos_l, edge_type, edge_time):
        # calculate the weights
        pos_i, pos_j = pos_k, pos_l
        spatial_dist = torch.norm(pos_j - pos_i, dim=-1)
        w_sym = torch.exp(-self.alpha * spatial_dist) # Symmetric weight based on spatial distance
        time_diff = torch.abs(edge_time)  # Use edge timestamps directly
        w_asym = torch.exp(-self.beta * time_diff) # Asymmetric weight based on temporal decay
        w = self.lambda_sym * w_sym + (1 - self.lambda_sym) * w_asym # Combine weights using lambda_sym

        # Relation-specific transformation
        edge_type = edge_type.long()  # Ensure edge_type is long for indexing
        out = torch.empty_like(x_j) # out: (num_edge, node_dim)

        for r in range(self.num_relations):
            mask = edge_type == r
            if mask.sum() > 0:  # Only process edges of type r
                edge_x0 = torch.cat([x_j[mask], x_i[mask]], dim=-1)
                print(edge_x0.shape)
                out[mask] = self.relation_mlps[r](edge_x0)

        return w.view(-1, 1) * out


    def update(self, aggr_out):
        # Placeholder for additional updates if needed
        return self.mlp(aggr_out)

