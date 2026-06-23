import numpy as np
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.data import Data, DataLoader
import argparse
import warnings
warnings.filterwarnings("ignore")
from model_geo import *
from GeoGData import *
from utils import *

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保 CuDNN 使用确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_args():
    '''
    Argument parser for running in command line
    '''
    parser = argparse.ArgumentParser('Geometric-Aware Graph Neural Network')
    # model par
    # task parameter
    parser.add_argument('--model_name', type=str, help='train model', default='gagnn')
    parser.add_argument('--gid', type=int, help='graph id', default=1)
    # training par
    parser.add_argument('--reg', type=float, help='hsic reg', default=1)

    parser.add_argument('--gpu', type=int, help='gpu', default=0)
    parser.add_argument('--n_epoch', type=int, help='number of epochs', default=100)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--spe', type=int, help='save per epoch', default=10)
    parser.add_argument('--seed', type=int, help='random seed', default=101)
    return parser.parse_args()


def train_eval_fold(data_base, train_idx, val_idx, test_idx, args, device, RunData):
    # 深拷贝数据并设置mask（避免污染原始数据）
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    data.val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx]     = True
    data.test_mask[test_idx]   = True

    # 初始化模型
    model = MultiRelationGNN(
        in_dim=RunData.num_features,
        out_dim=RunData.target_types,
        num_relations=RunData.edge_types,
        n=RunData.num_users,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    max_val_auc = -np.inf
    train_indices = torch.where(data.train_mask)[0]
    best_res = {}
    for epoch in range(args.n_epoch):
        model.train()
        model.training = True
        tr_tar = data.y[data.train_mask]                 # 所有训练标签
        total_loss = 0.0
        optimizer.zero_grad()
        out, hsic_loss = model(data.x, data.edge_index, data.edge_type)
        tr_pred = out[data.train_mask].squeeze(-1)
        cont_loss = contrastive_loss(tr_tar, tr_pred, device, m=3)
        loss = cont_loss + args.reg * hsic_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if epoch % args.spe == 0:
            model.eval()
            model.training = False
            with torch.no_grad():
                out, _ = model(data.x, data.edge_index, data.edge_type)

                # 验证集评估
                val_pred = out[data.val_mask].squeeze(-1)
                val_tar  = data.y[data.val_mask]
                val_auc = roc_auc_score(val_tar.cpu().numpy(), val_pred.cpu().numpy())

                # 若验证集AUC提升，则在测试集上评估并保存最佳结果
                if val_auc > max_val_auc:
                    max_val_auc = val_auc
                    # 测试集评估
                    ts_pred = out[data.test_mask].squeeze(-1)
                    ts_tar  = data.y[data.test_mask]
                    ts_auc = roc_auc_score(ts_tar.cpu().numpy(), ts_pred.cpu().numpy())
                    ts_auprc = average_precision_score(ts_tar.cpu().numpy(), ts_pred.cpu().numpy())

                    # 计算Recall@1（前2%阈值）
                    threshold = torch.quantile(ts_pred, 0.98, dim=None, keepdim=False)
                    ts_rec = transfer_pred(ts_pred, threshold)
                    class_rep = classification_report(ts_tar.cpu().numpy(), ts_rec.cpu().numpy(), output_dict=True)
                    ts_rec1 = class_rep['1']['recall']
                    ts_prec1 = class_rep['1']['precision']      # 异常类精确率（若需查看）
                    ts_f1 = class_rep['1']['f1-score']          # 异常类 F1（新增）

                    best_res['auc'], best_res['pr-auc'] = ts_auc, ts_auprc
                    best_res['rec'], best_res['prec'], best_res['f1'] = ts_rec1, ts_prec1, ts_f1

    return best_res


def main_cv(data, args, device, RunData):
    indices = np.arange(data.num_nodes)
    kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)  # 注意：args需要传入seed
    fold_res = {'auc':[],'pr-auc':[],'rec':[],'prec':[],'f1':[]}

    Fold_id = 0
    for train_val_idx, test_idx in kf.split(indices):
        # 从训练+验证索引中按 8:1 划分出训练集和验证集（验证集占剩余样本的1/9，即总样本的10%）
        val_size = len(train_val_idx) // 9   # 使得验证集占总样本约10%
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size, random_state=args.seed)

        # 转为torch张量并移到device
        train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
        val_idx   = torch.tensor(val_idx,   dtype=torch.long, device=device)
        test_idx  = torch.tensor(test_idx,  dtype=torch.long, device=device)

        res = train_eval_fold(data, train_idx, val_idx, test_idx, args, device, RunData)
        fold_res['auc'].append(res['auc'])
        fold_res['pr-auc'].append(res['pr-auc'])
        fold_res['rec'].append(res['rec'])
        fold_res['prec'].append(res['prec'])
        fold_res['f1'].append(res['f1'])
        print("Trained fold: ", Fold_id)
        Fold_id += 1
    return fold_res

if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    RunData = FinDGraphData(gid=args.gid)
    data = RunData.data.to(device)
    #RunData = SimulationData()
    #data = RunData.gen_simulation_data().to(device)

    Fold_res = main_cv(data, args, device, RunData)
    print(' AUC {:.4f}, '.format(np.mean(Fold_res['auc'])),
          ' REC-1 {:.4f}, '.format(np.mean(Fold_res['rec'])),
          ' PRAUC {:.4f}, '.format(np.mean(Fold_res['pr-auc'])),
          ' PREC-1 {:.4f}, '.format(np.mean(Fold_res['prec'])),
          ' F1-1 {:.4f}, '.format(np.mean(Fold_res['f1']))
    )