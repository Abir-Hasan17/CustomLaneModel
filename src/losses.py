import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(1,2,3)) + self.smooth
        den = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + self.smooth
        dice = 1 - num / den
        return bce + dice.mean()


class RowAnchorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        # Placeholder: since row-anchor labels depend on dataset conversion, we assume targets contain
        # 'exist' [B,L,R] and 'x_idx' [B,L,R] where x_idx in [0, X_bins) or -1 for missing
        exist_logits = outputs['row_exist_logits']
        x_logits = outputs['row_x_logits']
        exist_t = targets['exist'].float()
        loss_exist = F.binary_cross_entropy_with_logits(exist_logits, exist_t)
        # For regression/classification over x bins, apply CE only where exist==1
        B, L, R, X = x_logits.shape
        x_idx = targets['x_idx'].long().clamp(min=0)
        mask = (targets['x_idx'] >= 0).float()
        x_logits = x_logits.view(B*L*R, X)
        x_idx = x_idx.view(B*L*R)
        mask = mask.view(B*L*R)
        if mask.sum() > 0:
            loss_x = (F.cross_entropy(x_logits, x_idx, reduction='none') * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss_x = torch.tensor(0.0, device=exist_logits.device)
        return loss_exist + loss_x
