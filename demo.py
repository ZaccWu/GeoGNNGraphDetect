import torch
from torch_geometric.data import Data
import numpy as np
import torch.nn.functional as F
from model import *

# 1. 生成节点特征：1000个用户，每个用户有4个特征
num_users = 1000
num_features = 4
node_features = torch.rand((num_users, num_features), dtype=torch.float)  # 随机特征

# 2. 生成多种类型的有向边：3种关系
edge_types = 3
edges = []
edge_attrs = []

for edge_type_id in range(edge_types):
    num_edges = np.random.randint(2000, 5000)  # 每种关系随机生成边数量
    edge_index = torch.randint(0, num_users, (2, num_edges), dtype=torch.long)  # 随机边索引
    edge_attr = torch.ones((num_edges, 1), dtype=torch.int)*edge_type_id  # 边的类型label
    edges.append(edge_index)
    edge_attrs.append(edge_attr)

# 合并所有边及其属性
edge_index = torch.cat(edges, dim=1)
edge_attr = torch.cat(edge_attrs, dim=0)

# 3. 生成标签：用户兴趣类别（假设5类）
labels = torch.randint(0, 5, (num_users,), dtype=torch.long)

# 构建图数据对象
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
edge_types_tensor = torch.randint(0, edge_types, (edge_index.size(1),), dtype=torch.long)  # 边的类型

model = MultiRelationalGNN(
    in_channels=num_features,
    hidden_channels=16,
    out_channels=5,
    num_relations=edge_types
).to(device)

data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(1, 51):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

def test():
    model.eval()
    out = model(data.x, data.edge_index, data.edge_attr)
    _, pred = out.max(dim=1)
    acc = (pred == data.y).sum().item() / data.num_nodes
    print(f'Accuracy: {acc:.4f}')

test()