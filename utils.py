import torch
import torch.nn.functional as F

def contrastive_loss(target, pred_score, device, m=3):
    # target: 0-1, pred_score: float
    Rs = torch.mean(pred_score)
    delta = torch.std(pred_score)
    dev_score = (pred_score - Rs)/(delta + 1e-10)
    cont_score = torch.max(torch.zeros(pred_score.shape).to(device), m-dev_score)
    loss = dev_score[(1-target).nonzero()].mean()+cont_score[target.nonzero()].mean() 
    return loss # sum sample loss

def transfer_pred(pred_score, threshold):
    pred = pred_score.clone()
    pred[torch.where(pred_score < threshold)] = 0
    pred[torch.where(pred_score >= threshold)] = 1
    return pred

def class_orthloss(emb, labels):
    normal_mask = (labels == 0)
    anom_mask = (labels == 1)
    if normal_mask.sum() == 0 or anom_mask.sum() == 0:
        return torch.tensor(0.0, device=emb.device)

    H0 = F.normalize(emb[normal_mask].reshape(-1, emb.size(1)), dim=-1)  # [Nn, d]
    H1 = F.normalize(emb[anom_mask].reshape(-1, emb.size(1)), dim=-1)    # [Na, d]

    # 计算协方差矩阵（d, d）
    C_norm = H0.t() @ H0   # [d, d]
    C_anom = H1.t() @ H1  # [d, d]
    # F-norm 的平方 (除以元素数以消除样本量影响)
    co_loss = torch.trace(C_norm @ C_anom) / (H0.size(0) * H1.size(0))

    return co_loss