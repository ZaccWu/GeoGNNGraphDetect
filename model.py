import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class MultiRelationalGNN(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, aggr="mean"):
        """
        :param in_channels: 输入特征的维度
        :param out_channels: 输出特征的维度
        :param num_relations: 边的种类数（由 edge_attr 表示）
        :param aggr: 聚合方式（默认 "mean"）
        """
        super(MultiRelationalGNN, self).__init__(aggr=aggr)

        self.num_relations = num_relations

        # 为每种边类型定义不同的权重矩阵
        self.rel_weight = nn.ModuleList(
            [nn.Linear(hidden_channels, out_channels, bias=False) for _ in range(num_relations)]
        )

        # 目标节点特征转换
        self.node_weight = nn.Linear(in_channels, hidden_channels, bias=True)

    def forward(self, x, edge_index, edge_attr):
        """
        :param x: 节点特征矩阵 [num_nodes, in_channels]
        :param edge_index: 边索引 [2, num_edges]
        :param edge_attr: 边类型 [num_edges, 1]，每条边的类型（0 ~ num_relations-1）
        """
        # 输入特征转换
        x_trans = self.node_weight(x)  # 线性变换

        # 消息传递
        out = self.propagate(edge_index, x=x_trans, edge_attr=edge_attr)

        return out

    def message(self, x_j, edge_attr):
        """
        :param x_j: 发送节点的特征 [num_edges, out_channels]
        :param edge_attr: 边类型 [num_edges, 1]
        """
        # 将 edge_attr 转换为整型索引
        edge_type = edge_attr.squeeze().long()  # [num_edges]

        # 根据边类型选择对应的权重矩阵
        out = torch.zeros(len(x_j), 5) ## TODO:

        for rel in range(self.num_relations):
            mask = edge_type == rel  # 找到属于当前类型 rel 的边
            if mask.any():
                out[mask] = self.rel_weight[rel](x_j[mask])  # 应用 rel 对应的权重矩阵

        return out

    def update(self, aggr_out):
        """
        :param aggr_out: 聚合后的邻居消息 [num_nodes, out_channels]
        """
        # 直接返回聚合结果
        return aggr_out