import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import Linear

class MultiRelationGNN(MessagePassing):
    def __init__(self, node_dim, num_relations, aggr='add', lambda_sym=0.5, alpha=0.1, beta=0.1):
        super(MultiRelationGNN, self).__init__(aggr=aggr)
        self.num_relations = num_relations  # Number of relation types
        self.lambda_sym = lambda_sym  # Weight for symmetric part
        self.alpha = alpha  # Spatial decay parameter
        self.beta = beta  # Temporal decay parameter

        # Separate MLPs for each relation type
        self.relation_mlps = torch.nn.ModuleList(
            [Linear(node_dim * 2, node_dim) for _ in range(num_relations)]
        )
        self.mlp = Linear(node_dim, 5) # target num=5

    def forward(self, x, edge_index, edge_type, edge_time, pos):
    #def forward(self, x, edge_index):
        # x: Node features
        # edge_index: Edge indices (2, num_edges)
        # edge_type: Edge type information (int values)
        # edge_time: Edge timestamps
        # pos: Node positions (spatial positions)

        # row, col = edge_index  # row: source nodes, col: target nodes
        # # 从 pos 中提取源节点和目标节点的空间位置
        # pos_i = pos[row]  # Source node positions
        # pos_j = pos[col]  # Target node positions
        # edge_time = edge_time.view(-1,1)
        # num_nodes = x.size(0)

        return self.propagate(
            edge_index=edge_index, x=x,
            #edge_weight=edge_type,
            #edge_type=edge_type, edge_time=edge_time, pos_i=pos_i, pos_j=pos_j
        )

    # def message(self, x_j, x_i, edge_type, edge_time, pos_i, pos_j):
        # # Symmetric weight based on spatial distance
        # pos_i, pos_j = pos_i.view(-1,1), pos_j.view(-1,1)
        # spatial_dist = torch.norm(pos_j - pos_i, dim=-1)
        # w_sym = torch.exp(-self.alpha * spatial_dist)
        #
        # # Asymmetric weight based on temporal decay
        # time_diff = torch.abs(edge_time)  # Use edge timestamps directly
        # w_asym = torch.exp(-self.beta * time_diff)
        #
        # # Combine weights using lambda_sym
        # w = self.lambda_sym * w_sym + (1 - self.lambda_sym) * w_asym
        #
        # # Relation-specific transformation
        # edge_type = edge_type.long()  # Ensure edge_type is long for indexing
        # out = torch.empty_like(x_j)

        # for r in range(self.num_relations):
        #     mask = edge_type == r
        #     if mask.sum() > 0:  # Only process edges of type r
        #         out[mask] = self.relation_mlps[r](torch.cat([x_j[mask], x_i[mask]], dim=-1))
        #
        # #return w.view(-1, 1) * out
        # return out

    def message(self, x_j):
        # x_j 是源节点特征，自动根据 edge_index[0] 提取
        # edge_weight 是额外参数，需要显式传递
        #return edge_weight.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        # Placeholder for additional updates if needed
        return self.mlp(aggr_out)

