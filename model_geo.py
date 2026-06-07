from scipy.special import j1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv
from torch.autograd import Variable

## TODO 1: args function and repeated experiments
## TODO 2: compared with benchmarks and ablations
## TODO 3: explanable

class GeoMRGNNLayer(MessagePassing):
    def __init__(self, h_dim, num_relations):
        super().__init__()  # 聚合方法
        # Separate MLPs for each relation type
        self.relation_mlps = torch.nn.ModuleList(
            [nn.Linear(h_dim * 2, h_dim) for _ in range(num_relations)]
        )
        self.num_relations = num_relations  # Number of relation types

    def forward(self, x_emb, edge_index, edge_type):
        # x (num_nodes, node_feature_dim): Node features
        # edge_index (2, num_edges):  Edge indices
        # edge_type (num_edges, 1):   Edge type information
        return self.propagate(edge_index=edge_index, x=x_emb, edge_type=edge_type)

    def message(self, x_j, x_i, edge_type):
        # Relation-specific transformation
        edge_type = edge_type.long()  # Ensure edge_type is long for indexing
        msg_emb = torch.empty_like(x_j)  # msg_emb: (num_edge, h_dim)

        for r in range(self.num_relations):
            mask = edge_type == r
            mask = mask.squeeze(-1)
            if mask.sum() > 0:  # Only process edges of type r
                edge_x0 = torch.cat([x_j[mask], x_i[mask]], dim=-1)
                msg_emb[mask] = self.relation_mlps[r](edge_x0)
        out = msg_emb
        return out

    def update(self, aggr_out):
        # Placeholder for additional updates if needed
        return aggr_out



class MultiRelationGNN(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, n, h_dim=8):
        super().__init__()
        # layers
        self.field_mlp = nn.Linear(in_dim, h_dim)
        self.gat_conv = GATConv(in_dim, h_dim, add_self_loops=False)
        self.geonn_l1 = GeoMRGNNLayer(h_dim, num_relations)
        self.geonn_l2_1 = GeoMRGNNLayer(h_dim, num_relations)
        self.geonn_l2_2 = GeoMRGNNLayer(h_dim, num_relations)
        # binary classification with additive model
        self.out_mlp1 = nn.Sequential(nn.Linear(h_dim, out_dim),nn.LeakyReLU())
        self.out_mlp2 = nn.Sequential(nn.Linear(h_dim, out_dim),nn.LeakyReLU())
        self.out_mlp3 = nn.Sequential(nn.Linear(h_dim, out_dim),nn.LeakyReLU())
        self.out_mlp4 = nn.Sequential(nn.Linear(h_dim, out_dim),nn.LeakyReLU())
        self.out_all = nn.Sequential(nn.Linear(h_dim*4, out_dim),nn.LeakyReLU())

        self.n = n
        self.weight = nn.Parameter(torch.ones((self.n, 1)))
        for m in self.modules(): # weight initialization
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x, edge_index, edge_type):
        '''
        Construct different dimension featurs:
        1. node_emb0: raw feature of the focal node (after field transformation)
        2. node_nei1: aggr of neighbors' raw features (after transformation in gat)
        3. node_foc1: 1-layer geonn aggr
        4. node_foc2: 2-layer geonn aggr
        '''
        # 焦点特征
        node_focr = self.field_mlp(x)
        # 邻居特征（不区分边）
        node_nei1 = self.gat_conv(x, edge_index)
        # 一阶邻居特征（区分边）
        node_foc1 = self.geonn_l1(x_emb=node_focr, edge_index=edge_index, edge_type=edge_type)
        # 二阶邻居特征（区分边）
        node_emb1 = self.geonn_l2_1(x_emb=node_focr, edge_index=edge_index, edge_type=edge_type)
        node_foc2 = self.geonn_l2_1(x_emb=node_emb1, edge_index=edge_index, edge_type=edge_type)

        out = self.out_mlp1(node_focr) + self.out_mlp2(node_nei1) + self.out_mlp3(node_foc1) + self.out_mlp4(node_foc2) + self.out_all(torch.cat([node_focr, node_nei1, node_foc1, node_foc2], dim=-1))

        # TODO: 这里发现（1）GAT （2）用＋的方式（而不是concat+transform）（3）加一个总的交互Transformer效果更好
        # TODO: 想办法减少内存占用
        hsic_loss = Variable(torch.FloatTensor([0]).cuda())
        emb_list = [node_focr, node_nei1, node_foc1, node_foc2]
        for i in range(len(emb_list)):
            for j in range(i+1, len(emb_list)):
                hsic_loss += self.hsic(emb_list[i], emb_list[j], self.weight * self.weight, self.n).view(1) 

        return out, hsic_loss

    def hsic(self, emb1, emb2, sample_weights, dim):
        # R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
        # W1 = torch.mm(sample_weights, sample_weights.t())
        # K1 = W1*torch.mm(emb1, emb1.t())
        # K2 = W1*torch.mm(emb2, emb2.t())
        # RK1 = torch.mm(R, K1)
        # RK2 = torch.mm(R, K2)
        # HSIC = torch.trace(torch.mm(RK1, RK2))/((dim-1)*(dim-1))      

        K1 = torch.mm(emb1, emb1.t())
        K2 = torch.mm(emb2, emb2.t())
        HSIC = torch.trace(torch.mm(K1, K2))/((dim-1)*(dim-1))    
        return HSIC
