
import numpy as np
import torch.nn.functional as F
from model import *
from GeoGData import *
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

def set_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def contrastive_loss(target, pred_score, m=5):
    # target: 0-1, pred_score: float
    Rs = torch.mean(pred_score)
    delta = torch.std(pred_score)
    dev_score = (pred_score - Rs)/(delta + 1e-10)
    #print(dev_score.device, pred_score.device,target.device, Rs.device, delta.device)
    cont_score = torch.max(torch.zeros(pred_score.shape).to(device), m-dev_score)
    loss = dev_score[(1-target).nonzero()].sum()+cont_score[target.nonzero()].sum()
    return loss

def transfer_pred(pred_score, threshold):
    pred = pred_score.clone()
    pred[torch.where(pred_score < threshold)] = 0
    pred[torch.where(pred_score >= threshold)] = 1
    return pred

def main(data, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 100):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_type, data.edge_time, data.pos)
        tr_pred, tr_tar = out[data.train_mask].squeeze(-1), data.y[data.train_mask]
        loss = contrastive_loss(tr_tar, tr_pred)
        loss.backward()
        optimizer.step()

        # no batch training
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

            # model validation
            model.eval()
            out = model(data.x, data.edge_index, data.edge_type, data.edge_time, data.pos)
            val_pred, val_tar = out[data.val_mask].squeeze(-1), data.y[data.val_mask]

            val_r987 = transfer_pred(val_pred, torch.quantile(val_pred, 0.987, dim=None, keepdim=False))

            val_tar_list = val_tar.detach().cpu().numpy()
            val_pred_score_list = val_pred.detach().cpu().numpy()
            val_rec_list = val_r987.detach().cpu().numpy()

            print('Val Results: ')
            print(classification_report(np.array(val_tar_list), np.array(val_rec_list), digits=4))
            print("Val AUC: ", roc_auc_score(np.array(val_tar_list), np.array(val_rec_list)))


    # model test
    model.eval()
    out = model(data.x, data.edge_index, data.edge_type, data.edge_time, data.pos)
    ts_pred, ts_tar = out[data.test_mask].squeeze(-1), data.y[data.test_mask]

    ts_r987 = transfer_pred(ts_pred, torch.quantile(ts_pred, 0.987, dim=None, keepdim=False))

    ts_tar_list = ts_tar.detach().cpu().numpy()
    ts_pred_score_list = ts_pred.detach().cpu().numpy()
    ts_rec_list = ts_r987.detach().cpu().numpy()

    print('Test Results: ')
    print(classification_report(np.array(ts_tar_list), np.array(ts_rec_list), digits=4))
    print("Test AUC: ", roc_auc_score(np.array(ts_tar_list), np.array(ts_rec_list)))


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