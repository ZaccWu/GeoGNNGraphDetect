
import numpy as np
import torch.nn.functional as F
from model import *
from GeoGData import *
from sklearn.metrics import classification_report

def set_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


## TODO: write model training part
def main(data, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_type, data.edge_time, data.pos)
        tr_pred, tr_tar = out[data.train_mask], data.y[data.train_mask]
        loss = F.cross_entropy(tr_pred, tr_tar)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

            # model validation
            model.eval()
            out = model(data.x, data.edge_index, data.edge_type, data.edge_time, data.pos)
            _, out_pred = out.max(dim=1)

            val_pred, val_tar = out_pred[data.val_mask].detach().cpu().numpy(), data.y[data.val_mask].detach().cpu().numpy()

            print('Val Results: ')
            print(classification_report(np.array(val_tar), np.array(val_pred), digits=4))

    # model test
    model.eval()
    out = model(data.x, data.edge_index, data.edge_type, data.edge_time, data.pos)
    _, out_pred = out.max(dim=1)

    ts_pred, ts_tar = out_pred[data.test_mask].detach().cpu().numpy(), data.y[data.test_mask].detach().cpu().numpy()

    print('Test Results: ')
    print(classification_report(np.array(ts_tar), np.array(ts_pred), digits=4))



if __name__ == "__main__":
    seed = 101
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #RunData = SimulationData()
    #data = RunData.gen_simulation_data().to(device)

    RunData = FinDGraphData()
    data = RunData.data.to(device)
    model = MultiRelationGNN(
        in_dim=RunData.num_features,
        out_dim=RunData.target_types,
        num_relations=RunData.edge_types
    ).to(device)

    main(data, model)