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
    def __init__(self, in_dim, h_dim, num_relations):
        super().__init__()  # 聚合方法
        # Separate MLPs for each relation type
        self.h_dim = h_dim
        self.relation_mlps = torch.nn.ModuleList(
            [nn.Linear(in_dim * 2, self.h_dim) for _ in range(num_relations)]
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
        msg_emb = torch.empty(x_j.size(0), self.h_dim, device=x_j.device, dtype=x_j.dtype)

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
    def __init__(self, in_dim, out_dim, num_relations, n, h_dim=64):
        super().__init__()
        # layers
        self.feature_d = in_dim
        self.field_mlp = nn.Linear(in_dim, h_dim)

        self.gat_conv1 = GATConv(in_dim, h_dim, add_self_loops=False)
        self.gat_conv2 = GATConv(h_dim, h_dim, add_self_loops=False)

        self.geonn_l1 = GeoMRGNNLayer(in_dim, h_dim, num_relations)
        self.geonn_l2 = GeoMRGNNLayer(h_dim, h_dim, num_relations)

        # binary classification with additive model
        self.out_mlp1 = nn.Sequential(nn.Linear(h_dim, out_dim),nn.LeakyReLU())
        self.out_mlp2 = nn.Sequential(nn.Linear(h_dim, out_dim),nn.LeakyReLU())
        self.out_mlp3 = nn.Sequential(nn.Linear(h_dim, out_dim),nn.LeakyReLU())
        self.out_mlp4 = nn.Sequential(nn.Linear(h_dim, out_dim),nn.LeakyReLU())
        self.out_all = nn.Sequential(nn.Linear(h_dim*3, out_dim),nn.LeakyReLU())

        self.n = n
        self.training = True
        for m in self.modules(): # weight initialization
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x, edge_index, edge_type):
        # 焦点特征
        node_focr = self.field_mlp(x)

        # # 邻居特征（不区分边）
        # node_nei1 = self.gat_conv1(x, edge_index)
        # node_nei2 = self.gat_conv2(node_nei1, edge_index)

        # 一阶邻居特征（区分边）
        node_foc1 = self.geonn_l1(x_emb=x, edge_index=edge_index, edge_type=edge_type)
        node_foc2 = self.geonn_l2(x_emb=node_foc1, edge_index=edge_index, edge_type=edge_type)

        out = self.out_mlp1(node_focr) + self.out_mlp3(node_foc1) + self.out_mlp4(node_foc2) + self.out_all(torch.cat([node_focr, node_foc1, node_foc2], dim=-1))
        #out = self.out_mlp1(node_focr) + self.out_mlp2(node_nei2) + self.out_mlp3(node_foc1) + self.out_mlp4(node_foc2) + self.out_all(torch.cat([node_focr, node_nei1, node_foc1, node_foc2], dim=-1))
        emb_list = [node_focr, node_foc1, node_foc2]
        #emb_list = [node_focr, node_nei2, node_foc1, node_foc2]

        if self.training:
            hsic_loss = 0
            
            for i in range(len(emb_list)):
                for j in range(i+1, len(emb_list)):
                    hsic_loss += self.hsic_rff(emb_list[i], emb_list[j], self.feature_d).view(1) 
        else:
            hsic_loss = None
        return out, hsic_loss

    def rff_gaussian(self, x, gamma, n_features):
        """
        随机傅里叶特征近似高斯核（RBF）
        输入 x: [B, d]
        输出 phi: [B, n_features]
        """
        B, d = x.shape
        # 随机权重和偏置
        W = torch.randn(d, n_features, device=x.device) * torch.sqrt(torch.tensor(2.0 * gamma))
        b = 2 * torch.pi * torch.rand(n_features, device=x.device)
        z = x @ W + b
        phi = torch.sqrt(torch.tensor(2.0 / n_features)) * torch.cos(z)
        return phi

    def hsic_rff(self, x, y, n_features, gamma_x=None, gamma_y=None):
        """
        基于 RFF 的 HSIC 近似，内存 O(B * n_features + n_features^2)
        x, y: [B, d]
        """
        B = x.shape[0]
        # 自适应带宽：使用特征的标准差（启发式）
        if gamma_x is None:
            sigma_x = torch.median(torch.std(x, dim=0)).item()
            gamma_x = 1.0 / (2.0 * sigma_x**2 + 1e-8)
        if gamma_y is None:
            sigma_y = torch.median(torch.std(y, dim=0)).item()
            gamma_y = 1.0 / (2.0 * sigma_y**2 + 1e-8)

        phi_x = self.rff_gaussian(x, gamma_x, n_features)  # [B, D]
        phi_y = self.rff_gaussian(y, gamma_y, n_features)  # [B, D]

        # 中心化（减去均值）
        phi_x = phi_x - phi_x.mean(dim=0, keepdim=True)
        phi_y = phi_y - phi_y.mean(dim=0, keepdim=True)

        # 交叉协方差矩阵 (D, D)
        C = (phi_x.T @ phi_y) / (B - 1)
        hsic = torch.norm(C, p='fro') ** 2
        return hsic