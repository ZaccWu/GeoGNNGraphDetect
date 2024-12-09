from torch_geometric.nn import MessagePassing
import torch
from demo import SimulationData

class SimpleGNN(MessagePassing):
    def __init__(self):
        super(SimpleGNN, self).__init__(aggr='add')  # 使用加和聚合

    def forward(self, x, edge_index):
        # 打印输入数据的形状
        print("x shape:", x.shape)  # x: (num_nodes, feature_dim)
        print("edge_index shape:", edge_index.shape)  # edge_index: (2, num_edges)
        # 调用 propagate
        return self.propagate(edge_index=edge_index, x=x)

    def message(self, x_j):
        # 打印 x_j 的形状
        print("x_j shape:", x_j.shape)  # x_j: (num_edges, feature_dim)
        return x_j

    def update(self, aggr_out):
        # 聚合后的输出
        return aggr_out  # 这里可以做进一步的非线性变换，比如ReLU等



# 创建模型并执行
model = SimpleGNN()
SimData = SimulationData()
data =  SimData.gen_simulation_data()

out = model(data.x, data.edge_index)