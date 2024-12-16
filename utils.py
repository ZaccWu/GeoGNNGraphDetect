import torch

def contrastive_loss(target, pred_score, device, m=5):
    # target: 0-1, pred_score: float
    Rs = torch.mean(pred_score)
    delta = torch.std(pred_score)
    dev_score = (pred_score - Rs)/(delta + 1e-10)
    cont_score = torch.max(torch.zeros(pred_score.shape).to(device), m-dev_score)
    loss = dev_score[(1-target).nonzero()].sum()+cont_score[target.nonzero()].sum()
    return loss # sum sample loss

def transfer_pred(pred_score, threshold):
    pred = pred_score.clone()
    pred[torch.where(pred_score < threshold)] = 0
    pred[torch.where(pred_score >= threshold)] = 1
    return pred