import torch


def binary_f1_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    tp = (preds * targets).sum(dim=(1,2,3))
    fp = (preds * (1 - targets)).sum(dim=(1,2,3))
    fn = ((1 - preds) * targets).sum(dim=(1,2,3))
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1.mean().item()
