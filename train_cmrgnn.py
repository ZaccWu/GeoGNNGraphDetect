import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
import argparse
import warnings
warnings.filterwarnings("ignore")
from GeoGData import FinDGraphData
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
    parser = argparse.ArgumentParser('CMR-GNN for Fraud Detection')
    parser.add_argument('--gid', type=int, help='graph id', default=1)
    parser.add_argument('--gpu', type=int, help='gpu', default=0)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--spe', type=int, default=10)  # 用于验证频率（此处每epoch都验证）
    parser.add_argument('--seed', type=int, default=101)
    return parser.parse_args()

# ---------- 风险中心嵌入（仅使用度） ----------
class RiskCenterEmbedding(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.PReLU()
        )
        self.beta = torch.nn.Parameter(torch.ones(1))
        self.theta = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, degree):
        rho = self.beta * torch.log(degree + 1e-6) + self.theta
        return self.mlp(rho.unsqueeze(-1) * x)

# ---------- 多关系聚合（GAT + 自注意力融合） ----------
class MultiRelationalAggregator(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, heads=1):
        super().__init__()
        self.num_relations = num_relations
        self.gats = torch.nn.ModuleList([
            GATConv(in_dim, out_dim, heads=heads, concat=False) for _ in range(num_relations)
        ])
        self.fc_q = torch.nn.Linear(out_dim, out_dim)
        self.fc_k = torch.nn.Linear(out_dim, out_dim)

    def forward(self, x, edge_index, edge_type):
        h_list = []
        for r in range(self.num_relations):
            mask = (edge_type == r)
            if mask.sum() == 0:
                h = torch.zeros_like(x)
            else:
                h = self.gats[r](x, edge_index[:, mask])
            h_list.append(h)
        W = torch.stack(h_list, dim=1)           # [N, R, D]
        Q = self.fc_q(W)
        K = self.fc_k(W)
        attn = torch.softmax(torch.matmul(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5), dim=-1)
        W = torch.matmul(attn, W)
        return W.sum(dim=1)                      # [N, D]

# ---------- 组内平均增强 ----------
def group_average_enhance(emb, cluster_labels):
    # cluster_labels: torch.LongTensor, shape [N], values in [0, k-1]
    unique_labels = torch.unique(cluster_labels)
    means = torch.zeros((len(unique_labels), emb.size(1)), device=emb.device)
    for i, lab in enumerate(unique_labels):
        mask = (cluster_labels == lab)
        means[i] = emb[mask].mean(dim=0)
    # 直接利用 labels 作为索引（需确保 labels 是 LongTensor 且连续）
    return means[cluster_labels]   # 完全向量化，无 Python 循环遍历 N

# ---------- 可学习的加权融合（替代 GRU） ----------
class WeightedFusion(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, h_agg, h_enhanced):
        alpha = torch.sigmoid(self.weight)
        return alpha * h_enhanced + (1 - alpha) * h_agg

# ---------- 主模型（优化版） ----------
class CMRGNN_Optimized(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, num_clusters, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.num_clusters = num_clusters
        self.rce = RiskCenterEmbedding(in_dim, hidden_dim)
        self.aggregators = torch.nn.ModuleList()
        self.fusions = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.aggregators.append(MultiRelationalAggregator(hidden_dim, hidden_dim, num_relations))
            self.fusions.append(WeightedFusion(hidden_dim))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )
        self.cached_labels = None
        self.update_interval = 10   # 每10轮更新聚类

    def forward(self, x, edge_index, edge_type, degree, epoch=None):
        x = self.rce(x, degree)
        H = x
        for l in range(self.num_layers):
            H_agg = self.aggregators[l](H, edge_index, edge_type)

            # 聚类（根据epoch决定是否更新）
            if epoch is not None and (epoch % self.update_interval == 0 or self.cached_labels is None):
                with torch.no_grad():
                    emb_np = H_agg.detach().cpu().numpy()
                    kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=0,
                                             batch_size=1000, n_init=3)
                    labels = kmeans.fit_predict(emb_np)
                    self.cached_labels = torch.tensor(labels, device=H_agg.device)

            if self.cached_labels is not None:
                labels = self.cached_labels
            else:
                # 首次前向时若无缓存，临时聚类（通常训练时一定会先缓存）
                with torch.no_grad():
                    emb_np = H_agg.detach().cpu().numpy()
                    kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=0,
                                             batch_size=1000, n_init=3)
                    labels = torch.tensor(kmeans.fit_predict(emb_np), device=H_agg.device)

            H_enhanced = group_average_enhance(H_agg, labels)
            H = self.fusions[l](H_agg, H_enhanced)

        return self.mlp(H)

# ---------- 训练函数（带早停） ----------
def train_eval_fold(data, train_idx, val_idx, test_idx, args, device, RunData):
    # 构造mask
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    data.val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx]     = True
    data.test_mask[test_idx]   = True

    in_dim = data.x.size(1)
    hidden_dim = 32   # 降低维度加速
    num_relations = RunData.edge_types
    num_clusters = 5 if args.gid == 1 else 10   # 根据数据集调整
    model = CMRGNN_Optimized(in_dim, hidden_dim, 2, num_relations, num_clusters).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 类别权重
    y = data.y.cpu().numpy()
    pos_weight = (len(y) - y.sum()) / y.sum()
    weight = torch.tensor([1.0, pos_weight], dtype=torch.float, device=device)

    best_val_auc = -np.inf
    best_res = {}

    # 计算节点度
    deg = torch.zeros(data.num_nodes, dtype=torch.float, device=device)
    deg.scatter_add_(0, data.edge_index[0], torch.ones(data.edge_index.size(1), dtype=torch.float, device=device))

    for epoch in range(args.n_epoch):
        #print(epoch)
        model.train()
        optimizer.zero_grad()


        logits = model(data.x, data.edge_index, data.edge_type, deg, epoch=epoch)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask], weight=weight)
        loss.backward()
        optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            logits_val = model(data.x, data.edge_index, data.edge_type, deg, epoch=None)   # 不更新聚类
            val_logits = logits_val[data.val_mask]
            val_prob = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            val_label = data.y[data.val_mask].cpu().numpy()
            val_auc = roc_auc_score(val_label, val_prob)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                no_improve = 0
                # 测试集评估
                test_logits = logits_val[data.test_mask]
                test_prob = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
                test_label = data.y[data.test_mask].cpu().numpy()
                ts_auc = roc_auc_score(test_label, test_prob)
                ts_auprc = average_precision_score(test_label, test_prob)
                th = np.quantile(test_prob, 0.98)
                pred_bin = (test_prob >= th).astype(int)
                rep = classification_report(test_label, pred_bin, output_dict=True, zero_division=0)
                best_res = {
                    'auc': ts_auc,
                    'pr-auc': ts_auprc,
                    'rec': rep['1']['recall'] if '1' in rep else 0.0,
                    'prec': rep['1']['precision'] if '1' in rep else 0.0,
                    'f1': rep['1']['f1-score'] if '1' in rep else 0.0
                }

    return best_res

# ---------- 10折交叉验证 ----------
def main_cv(data, args, device, RunData):
    indices = np.arange(data.num_nodes)
    kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)
    fold_res = {'auc': [], 'pr-auc': [], 'rec': [], 'prec': [], 'f1': []}
    for fold_id, (train_val_idx, test_idx) in enumerate(kf.split(indices)):
        val_size = len(train_val_idx) // 9
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size, random_state=args.seed)
        train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
        val_idx   = torch.tensor(val_idx, dtype=torch.long, device=device)
        test_idx  = torch.tensor(test_idx, dtype=torch.long, device=device)
        res = train_eval_fold(data, train_idx, val_idx, test_idx, args, device, RunData)
        for k in fold_res:
            fold_res[k].append(res.get(k, 0.0))
        print(f"Fold {fold_id} done.")
    return fold_res


if __name__ == "__main__":
    args = get_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)   # 确保已定义
    RunData = FinDGraphData(gid=args.gid)
    data = RunData.data.to(device)
    fold_res = main_cv(data, args, device, RunData)
    print("Model: CMR-GNN, data gid:", args.gid)
    print(' AUC {:.4f}, REC-1 {:.4f}, PRAUC {:.4f}, PREC-1 {:.4f}, F1-1 {:.4f}'.format(
        np.mean(fold_res['auc']), np.mean(fold_res['rec']),
        np.mean(fold_res['pr-auc']), np.mean(fold_res['prec']),
        np.mean(fold_res['f1'])))