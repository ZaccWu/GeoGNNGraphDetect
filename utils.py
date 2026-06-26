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

# def alignment_loss(proj_list, labels): ## TODO：目前这个alignment loss似乎没什么用，可以去掉
#     normal_mask = (labels == 0)
#     if normal_mask.sum() == 0:
#         return torch.tensor(0.0)
#     loss = 0.0
#     count = 0
#     for i in range(len(proj_list)):
#         for j in range(i+1, len(proj_list)):
#             zi = F.normalize(proj_list[i][normal_mask], dim=-1)  # 确保归一化
#             zj = F.normalize(proj_list[j][normal_mask], dim=-1)
#             # 使用平方欧氏距离（天然非负）
#             # loss += ((zi - zj)**2).sum(dim=-1).mean()
#             # 或 1 - cos
#             cos_sim = (zi * zj).sum(dim=-1)
#             loss += (1 - cos_sim).mean()
#             count += 1
#     return loss / count if count > 0 else torch.tensor(0.0)