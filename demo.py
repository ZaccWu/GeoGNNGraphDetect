
import numpy as np
import torch.nn.functional as F
from model import *
from GeoGData import *

def set_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)



def train(data, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_type, data.edge_time, data.pos)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
    return model

def test(model):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_type, data.edge_time, data.pos)
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
        in_dim=SimData.num_features,
        out_dim=SimData.target_types,
        num_relations=SimData.edge_types
    ).to(device)

    model = train(data, model)
    test(model)