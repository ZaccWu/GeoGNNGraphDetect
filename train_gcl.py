import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch_geometric.utils import dropout_adj, mask_feature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
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
    # task parameter# rd，gat, gin, rgat, egc
    parser.add_argument('--model_name', type=str, help='train model', default='gcl') 
    parser.add_argument('--gid', type=int, help='graph id', default=1)

    parser.add_argument('--gpu', type=int, help='gpu', default=0)
    parser.add_argument('--n_epoch', type=int, help='number of epochs', default=100)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--spe', type=int, help='save per epoch', default=10)
    parser.add_argument('--seed', type=int, help='random seed', default=101)
    return parser.parse_args()

# base model
class GAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, h_dim=32):
        super().__init__()
        self.gatconv_1 = GATConv(in_dim, h_dim)
        self.gatconv_2 = GATConv(h_dim, out_dim)

    def forward(self, x, edge_index, edge_type):
        x1 = self.gatconv_1(x, edge_index)
        x2 = self.gatconv_2(x1, edge_index)
        return x2


# ---------- 数据增强 ----------
def graph_augment(data, feat_mask_rate=0.2, edge_drop_rate=0.2):
    """
    对单个图进行随机增强，返回新图（保留原始结构副本）
    """
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    edge_type = data.edge_type.clone() if hasattr(data, 'edge_type') else None

    # 特征掩蔽：随机将特征置零
    if feat_mask_rate > 0:
        mask = torch.rand(x.size(1), device=x.device) < feat_mask_rate
        x[:, mask] = 0

    # 边删除
    if edge_drop_rate > 0 and edge_index.size(1) > 0:
        edge_index, edge_type = dropout_adj(edge_index, edge_type, p=edge_drop_rate, force_undirected=False)

    # 构建新 Data 对象（保持原数据属性）
    aug_data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=data.y, num_nodes=data.num_nodes)
    return aug_data

# ---------- 投影头 ----------
class ProjectionHead(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

# ---------- 对比损失（InfoNCE）----------
def info_nce_loss(z1, z2, temperature=0.5, n_neg=128):
    """
    z1, z2: [N, D] 两个视图的表示
    n_neg: 每个正样本对应的负样本数量
    """
    N = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # 正样本相似度 (对角线)
    pos_sim = (z1 * z2).sum(dim=-1) / temperature  # [N]

    # 随机采样负样本（从 z2 中抽取，也可从 z1+z2 中混合）
    # 如果 N 小于 n_neg，则用所有节点作为负样本（但仍避免过大的矩阵）
    n_neg = min(n_neg, N-1)  # 确保负样本数不超过 N-1
    if n_neg <= 0:
        # 如果只有1个节点，无法有负样本，则返回0损失
        return torch.tensor(0.0, device=z1.device)

    # 随机选择负样本索引（避免采样到自身正样本）
    # 这里简化：直接随机从所有节点中采样，可能会采样到同节点（但概率低），可忽略
    neg_idx = torch.randint(0, N, (N, n_neg), device=z1.device)

    # 计算负样本相似度 (z1 与 z2[neg_idx] 的点积)
    # 使用 bmm 实现批量矩阵乘法 [N, 1, D] x [N, D, n_neg] -> [N, n_neg]
    neg_sim = torch.bmm(z1.unsqueeze(1), z2[neg_idx].permute(0, 2, 1)).squeeze(1) / temperature  # [N, n_neg]

    # 拼接正负样本
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [N, 1+n_neg]
    labels = torch.zeros(N, dtype=torch.long, device=z1.device)  # 正样本索引为0

    loss = F.cross_entropy(logits, labels)
    return loss



# ---------- 训练函数（每个fold内）----------
def train_eval_fold_gcl(data, train_idx, val_idx, test_idx, args, device, RunData):
    # 构造 mask（仅用于后续评估，训练时不用标签）
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    data.val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx]     = True
    data.test_mask[test_idx]   = True

    outemb_dim = 8  # 可配置
    encoder = GAT(in_dim=RunData.num_features, out_dim=outemb_dim, num_relations=RunData.edge_types).to(device)
    proj_head = ProjectionHead(in_dim=outemb_dim, hidden_dim=outemb_dim, out_dim=outemb_dim).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(proj_head.parameters()), lr=args.lr)

    best_val_auc = -np.inf
    best_res = {}

    # 预训练循环（无监督）
    for epoch in range(args.n_epoch):
        encoder.train()
        proj_head.train()

        # 生成两个增强视图
        aug1 = graph_augment(data, feat_mask_rate=0.2, edge_drop_rate=0.2)
        aug2 = graph_augment(data, feat_mask_rate=0.2, edge_drop_rate=0.2)

        # 前向传播（使用全部节点）
        h1 = encoder(aug1.x, aug1.edge_index, aug1.edge_type)  # [N, emb_dim]
        h2 = encoder(aug2.x, aug2.edge_index, aug2.edge_type)
        z1 = proj_head(h1)
        z2 = proj_head(h2)

        loss = info_nce_loss(z1, z2, temperature=0.5)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证（每 spe 个 epoch 评估下游性能，需训练一个临时分类器）
        if epoch % args.spe == 0:
            encoder.eval()
            with torch.no_grad():
                # 获取训练集和验证集的表示
                train_emb = encoder(data.x, data.edge_index, data.edge_type)[data.train_mask].cpu().numpy()
                val_emb = encoder(data.x, data.edge_index, data.edge_type)[data.val_mask].cpu().numpy()
                train_label = data.y[data.train_mask].cpu().numpy()
                val_label = data.y[data.val_mask].cpu().numpy()

                # 训练逻辑回归（L2正则）
                clf = LogisticRegression(max_iter=1000, C=1.0, random_state=args.seed)
                clf.fit(train_emb, train_label)
                val_pred_prob = clf.predict_proba(val_emb)[:, 1]
                val_auc = roc_auc_score(val_label, val_pred_prob)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    # 测试集评估
                    test_emb = encoder(data.x, data.edge_index, data.edge_type)[data.test_mask].cpu().numpy()
                    test_label = data.y[data.test_mask].cpu().numpy()
                    test_pred_prob = clf.predict_proba(test_emb)[:, 1]
                    ts_auc = roc_auc_score(test_label, test_pred_prob)
                    ts_auprc = average_precision_score(test_label, test_pred_prob)
                    # Recall@1（取98%分位数）
                    threshold = np.quantile(test_pred_prob, 0.98)
                    ts_pred_bin = (test_pred_prob >= threshold).astype(int)
                    from sklearn.metrics import classification_report
                    rep = classification_report(test_label, ts_pred_bin, output_dict=True)
                    ts_rec1 = rep['1']['recall']
                    ts_prec1 = rep['1']['precision']
                    ts_f1 = rep['1']['f1-score']
                    best_res = {'auc':ts_auc, 'pr-auc':ts_auprc, 'rec':ts_rec1, 'prec':ts_prec1, 'f1':ts_f1}

    return best_res

# ---------- 主函数（与原保持一致）----------
def main_cv(data, args, device, RunData):
    indices = np.arange(data.num_nodes)
    kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)
    fold_res = {'auc':[], 'pr-auc':[], 'rec':[], 'prec':[], 'f1':[]}
    for fold_id, (train_val_idx, test_idx) in enumerate(kf.split(indices)):
        val_size = len(train_val_idx) // 9
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size, random_state=args.seed)
        train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
        val_idx   = torch.tensor(val_idx, dtype=torch.long, device=device)
        test_idx  = torch.tensor(test_idx, dtype=torch.long, device=device)
        res = train_eval_fold_gcl(data, train_idx, val_idx, test_idx, args, device, RunData)
        for k in fold_res:
            fold_res[k].append(res.get(k, 0.0))
        print(f"Fold {fold_id} done.")
    return fold_res

if __name__ == "__main__":
    args = get_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    RunData = FinDGraphData(gid=args.gid)
    data = RunData.data.to(device)
    Fold_res = main_cv(data, args, device, RunData)
    print("Model, data:", args.model_name, args.gid)
    print(' AUC {:.4f}, REC-1 {:.4f}, PRAUC {:.4f}, PREC-1 {:.4f}, F1-1 {:.4f}'.format(
        np.mean(Fold_res['auc']), np.mean(Fold_res['rec']), np.mean(Fold_res['pr-auc']),
        np.mean(Fold_res['prec']), np.mean(Fold_res['f1'])))