import numpy as np
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data, DataLoader
import argparse
import warnings
warnings.filterwarnings("ignore")

from model import *
from GeoGData import *
from utils import *



def get_args():
    '''
    Argument parser for running in command line
    '''
    parser = argparse.ArgumentParser('Geometric-Aware Graph Neural Network')
    # model par
    # task parameter
    parser.add_argument('--model_name', type=str, help='train model', default='gagnn')

    # training par
    parser.add_argument('--gpu', type=int, help='gpu', default=0)
    parser.add_argument('--n_epoch', type=int, help='number of epochs', default=100)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--bs', type=int, help='batch size', default=262144)  # node-level batch
    parser.add_argument('--spe', type=int, help='save per epoch', default=10)
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    # load data
    RunData = FinDGraphData()
    data = RunData.data.to(device)
    #RunData = SimulationData()
    #data = RunData.gen_simulation_data().to(device)

    # load model
    model = MultiRelationGNN(
        in_dim=RunData.num_features,
        out_dim=RunData.target_types,
        num_relations=RunData.edge_types
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    max_val_metrics = -np.inf
    for epoch in range(args.n_epoch):
        model.train()
        # design for minibatch training
        tr_tar = data.y[data.train_mask]
        n_iter = int(len(tr_tar)/args.bs)
        total_loss = 0
        for iter in range(n_iter+1):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_type, data.edge_time)
            tr_pred = out[data.train_mask].squeeze(-1)

            if (iter+1)*args.bs >= len(tr_tar):
                tr_pred_batch, tr_tar_batch = tr_pred[iter*args.bs: ], tr_pred[iter*args.bs: ]
            else:
                tr_pred_batch, tr_tar_batch = tr_pred[iter*args.bs: (iter+1)*args.bs], tr_tar[iter*args.bs: (iter+1)*args.bs]

            loss = contrastive_loss(tr_tar_batch, tr_pred_batch, device, m=5)
            loss.backward()
            optimizer.step()
            total_loss+=loss

        # select models with AUC
        if epoch % args.spe == 0:
            # model validation
            model.eval()
            out = model(data.x, data.edge_index, data.edge_type, data.edge_time)
            val_pred, val_tar = out[data.val_mask].squeeze(-1), data.y[data.val_mask]

            val_r987 = transfer_pred(val_pred, torch.quantile(val_pred, 0.987, dim=None, keepdim=False))

            val_tar_list = val_tar.detach().cpu().numpy()
            val_pred_score_list = val_pred.detach().cpu().numpy()
            val_rec_list = val_r987.detach().cpu().numpy()

            val_class_rep = classification_report(np.array(val_tar_list), np.array(val_rec_list), output_dict=True)
            val_auc = roc_auc_score(np.array(val_tar_list), np.array(val_pred_score_list))

            if val_auc > max_val_metrics:
                max_val_metrics = val_auc
                # model test
                model.eval()
                out = model(data.x, data.edge_index, data.edge_type, data.edge_time)
                ts_pred, ts_tar = out[data.test_mask].squeeze(-1), data.y[data.test_mask]

                ts_r987 = transfer_pred(ts_pred, torch.quantile(ts_pred, 0.987, dim=None, keepdim=False))

                ts_tar_list = ts_tar.detach().cpu().numpy()
                ts_pred_score_list = ts_pred.detach().cpu().numpy()
                ts_rec_list = ts_r987.detach().cpu().numpy()

                ts_class_rep = classification_report(np.array(ts_tar_list), np.array(ts_rec_list), output_dict=True)
                ts_auc = roc_auc_score(np.array(ts_tar_list), np.array(ts_pred_score_list))

                Rep_ts_auc, Rep_ts_rec1 = ts_auc, ts_class_rep['1']['recall']

    return Rep_ts_auc, Rep_ts_rec1



if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    Rep_res = {'seed': [], 'AUC': [], 'rec1': []}  # all possible evaluation metrics
    for seed in range(1, 11):
        set_seed(seed)
        Rep_ts_auc, Rep_ts_rec1 = main()
        Rep_res['seed'].append(seed)
        Rep_res['AUC'].append(Rep_ts_auc)
        Rep_res['rec1'].append(Rep_ts_rec1)

        print(seed, ' AUC {:.4f}, '.format(Rep_ts_auc),
              'Rec1 {:.4f}, '.format(Rep_ts_rec1))

    print(' AUC_ALL {:.4f} ({:.4f}), '.format(np.mean(Rep_res['AUC']), np.std(Rep_res['AUC'])),
          ' REC1_ALL {:.4f} ({:.4f}), '.format(np.mean(Rep_res['rec1']), np.std(Rep_res['rec1']))
          )