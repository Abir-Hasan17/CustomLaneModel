import torch
import torch.nn as nn
import torch.nn.functional as F
from .scnn_model import SCNNSegModel
from .uflf_model import UFLDLikeModel


class HybridLaneModel(nn.Module):
    def __init__(self, backbone: str = 'resnet18', num_rows: int = 18, num_lanes: int = 4, x_bins: int = 200):
        super().__init__()
        self.seg_model = SCNNSegModel(backbone=backbone)
        self.row_model = UFLDLikeModel(backbone=backbone, num_rows=num_rows, num_lanes=num_lanes, x_bins=x_bins)

    def forward(self, x):
        seg = self.seg_model(x)
        row = self.row_model(x)
        return {**seg, **row}
