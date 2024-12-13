import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

BINS = 3

class GeoMRGNNLayer(MessagePassing):
    def __init__(self, h_dim, num_relations, lambda_sym, beta):
        super().__init__()  # 聚合方法
        # Separate MLPs for each relation type
        self.relation_mlps = torch.nn.ModuleList(
            [nn.Linear(h_dim * 2, h_dim) for _ in range(num_relations)]
        )
        self.num_relations = num_relations  # Number of relation types
        self.lambda_sym = lambda_sym  # Weight for symmetric part
        self.beta = beta  # Temporal decay parameter
        ## TODO: found that uniform weights (rather than learnable weights) for beta are better

    def forward(self, x_emb, edge_index, edge_type, edge_time):
        # x (num_nodes, node_feature_dim): Node features
        # edge_index (2, num_edges):  Edge indices
        # edge_type (num_edges, 1):   Edge type information
        # edge_time (num_edges, num_time_features):      Edge time features
        # pos (num_nodes, pos_feature_dim): Node spatial positions
        return self.propagate(edge_index=edge_index, x=x_emb, edge_type=edge_type, edge_time=edge_time)

    def message(self, x_j, x_i, edge_type, edge_time):
        num_edge_src, tdiff_edge_src = edge_time[:, 0:BINS*1], edge_time[:, BINS*1:BINS*2]
        num_edge_trg, tdiff_edge_trg = edge_time[:, BINS*2:BINS*3], edge_time[:, BINS*3:BINS*4]
        logit = torch.matmul(self.beta[0:BINS*1].T, num_edge_src.T)  + torch.matmul(self.beta[BINS*1:BINS*2].T, tdiff_edge_src.T) + torch.matmul(self.beta[BINS*2:BINS*3].T, num_edge_trg.T) + torch.matmul(self.beta[BINS*3:BINS*4].T, tdiff_edge_trg.T)
        logit = logit.squeeze(0)

        w = self.lambda_sym * torch.exp(-logit)  # (num_edges)

        # Relation-specific transformation
        edge_type = edge_type.long()  # Ensure edge_type is long for indexing
        msg_emb = torch.empty_like(x_j)  # msg_emb: (num_edge, h_dim)

        for r in range(self.num_relations):
            mask = edge_type == r
            mask = mask.squeeze(-1)
            if mask.sum() > 0:  # Only process edges of type r
                edge_x0 = torch.cat([x_j[mask], x_i[mask]], dim=-1)
                msg_emb[mask] = self.relation_mlps[r](edge_x0)

        out = w.view(-1, 1) * msg_emb
        #out = msg_emb  # no weights
        return out

    def update(self, aggr_out):
        # Placeholder for additional updates if needed
        return aggr_out



class MultiRelationGNN(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, h_dim=32):
        super().__init__()
        # learnable weights
        self.lambda_sym = nn.Parameter(torch.empty(1, 1))
        self.beta = nn.Parameter(torch.empty(4*BINS, 1))

        self.reset_parameters()

        # layers
        self.field_mlp = nn.Linear(in_dim, h_dim)
        self.geonn_l1 = GeoMRGNNLayer(h_dim, num_relations, self.lambda_sym, self.beta)
        self.geonn_l2 = GeoMRGNNLayer(h_dim, num_relations, self.lambda_sym, self.beta)
        # binary classification with additive model
        self.out_mlp0 = nn.Sequential(
            nn.Linear(h_dim, out_dim),
            nn.LeakyReLU(),
        )
        self.out_mlp1 = nn.Sequential(
            nn.Linear(h_dim, out_dim),
            nn.LeakyReLU(),
        )
        self.out_mlp2 = nn.Sequential(
            nn.Linear(h_dim, out_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x, edge_index, edge_type, edge_time):
        node_emb0 = self.field_mlp(x)
        node_emb1 = self.geonn_l1(x_emb=node_emb0, edge_index=edge_index, edge_type=edge_type,
                                  edge_time=edge_time)
        node_emb2 = self.geonn_l2(x_emb=node_emb1, edge_index=edge_index, edge_type=edge_type,
                                  edge_time=edge_time)

        out = self.out_mlp2(node_emb2) + self.out_mlp1(node_emb1) + self.out_mlp0(node_emb0)
        return out

    def reset_parameters(self):
        nn.init.xavier_normal_(self.lambda_sym)
        nn.init.xavier_normal_(self.beta)


