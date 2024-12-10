import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd

class SimulationData():
    def __init__(self):
        super(SimulationData, self).__init__()
        self.num_users = 1000
        self.num_features = 4
        self.edge_types = 5
        self.target_types = 3

    def gen_simulation_data(self):
        node_features = torch.rand((self.num_users, self.num_features), dtype=torch.float)  # 随机特征
        edges, edge_attrs = [], []
        for edge_type_id in range(self.edge_types):
            num_edges = np.random.randint(2000, 5000)  # 每种关系随机生成边数量
            edge_index = torch.randint(0, self.num_users, (2, num_edges), dtype=torch.long)  # 随机边索引
            edge_type = torch.ones((num_edges, 1), dtype=torch.int)*edge_type_id  # 边的类型label
            edges.append(edge_index)
            edge_attrs.append(edge_type)
        # concat all types of edges
        edge_index, edge_type = torch.cat(edges, dim=1), torch.cat(edge_attrs, dim=0)
        # 生成边时间
        edge_time = torch.randint(0, 100, (edge_index.shape[1],))  # Edge timestamps (e.g., formation time)
        # Random node positions (spatial features)
        pos = torch.randn(self.num_users, 3)  # 3D positions

        # 3. 生成标签：用户兴趣类别（假设3类）
        labels = torch.randint(0, self.target_types, (self.num_users,), dtype=torch.long)
        # 构建图数据对象
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_time=edge_time,
            pos=pos,
            y=labels)
        return data


class FinDGraphData():
    def __init__(self):
        super(FinDGraphData, self).__init__()
        data_raw = np.load('./data/DGraphFin/dgraphfin.npz')
        self.num_users = data_raw['x'].shape[0]
        self.num_features = data_raw['x'].shape[-1]
        self.edge_types = len(pd.Series(data_raw['edge_type']).value_counts())
        self.target_types = 2

        self.data = Data(
            x=torch.FloatTensor(data_raw['x']),
            edge_index=torch.LongTensor(data_raw['edge_index']).T, # -> (2, num_edges)
            edge_type=torch.LongTensor(data_raw['edge_type']), # (num_edges,)
            edge_time=torch.FloatTensor(data_raw['edge_timestamp']), # (num_edges,)
            pos=None,
            y=torch.LongTensor(data_raw['y']),
            train_mask=torch.LongTensor(data_raw['train_mask']),
            val_mask=torch.LongTensor(data_raw['valid_mask']),
            test_mask=torch.LongTensor(data_raw['test_mask'])
        )
        del data_raw




