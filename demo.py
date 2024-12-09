import torch
from torch_geometric.data import Data
import numpy as np
import torch.nn.functional as F
from model import *

def set_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

class SimulationData():
    def __init__(self):
        super(SimulationData, self).__init__()
        self.num_users = 1000
        self.num_features = 4
        self.edge_types = 5

    def gen_simulation_data(self):
        node_features = torch.rand((self.num_users, self.num_features), dtype=torch.float)  # 随机特征
        edges, edge_attrs = [], []
        for edge_type_id in range(self.edge_types):
            num_edges = np.random.randint(2000, 5000)  # 每种关系随机生成边数量
            edge_index = torch.randint(0, self.num_users, (2, num_edges), dtype=torch.long)  # 随机边索引
            edge_attr = torch.ones((num_edges, 1), dtype=torch.int)*edge_type_id  # 边的类型label
            edges.append(edge_index)
            edge_attrs.append(edge_attr)
        # concat all types of edges
        edge_index, edge_attr = torch.cat(edges, dim=1), torch.cat(edge_attrs, dim=0)
        # 生成边时间
        edge_time = torch.randint(0, 100, (edge_index.shape[1],))  # Edge timestamps (e.g., formation time)
        # Random node positions (spatial features)
        pos = torch.randn(self.num_users, 3)  # 3D positions

        # 3. 生成标签：用户兴趣类别（假设5类）
        labels = torch.randint(0, 5, (self.num_users,), dtype=torch.long)

        # 构建图数据对象
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_time=edge_time,
            pos=pos,
            y=labels)
        return data


def train(data, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.edge_time, data.pos)
        #out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
    return model

def test(model):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_attr, data.edge_time, data.pos)
    #out = model(data.x, data.edge_index)
    _, pred = out.max(dim=1)
    acc = (pred == data.y).sum().item() / data.num_nodes
    print(f'Accuracy: {acc:.4f}')


if __name__ == "__main__":
    seed = 101
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SimData = SimulationData()
    data =  SimData.gen_simulation_data().to(device)
    model = MultiRelationGNN(
        node_dim=SimData.num_features,
        num_relations=SimData.edge_types
    ).to(device)

    model = train(data, model)
    test(model)