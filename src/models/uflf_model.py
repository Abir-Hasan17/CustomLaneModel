import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class RowAnchorHead(nn.Module):
    def __init__(self, in_ch: int, num_rows: int, num_lanes: int, x_bins: int = 200):
        super().__init__()
        self.num_rows = num_rows
        self.num_lanes = num_lanes
        self.x_bins = x_bins
        self.cls = nn.Linear(in_ch, num_rows * num_lanes)
        self.reg = nn.Linear(in_ch, num_rows * num_lanes * x_bins)

    def forward(self, feats):
        # Global pooling
        x = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        cls_logits = self.cls(x)  # [B, R*L]
        reg_logits = self.reg(x)  # [B, R*L*X]
        return {
            "row_exist_logits": cls_logits.view(-1, self.num_lanes, self.num_rows),
            "row_x_logits": reg_logits.view(-1, self.num_lanes, self.num_rows, self.x_bins)
        }


class UFLDLikeModel(nn.Module):
    def __init__(self, backbone: str = 'resnet18', num_rows: int = 18, num_lanes: int = 4, x_bins: int = 200):
        super().__init__()
        net = resnet18(weights=None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.head = RowAnchorHead(512, num_rows, num_lanes, x_bins)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)
