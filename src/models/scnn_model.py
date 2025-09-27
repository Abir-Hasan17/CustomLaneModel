import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34


class SCNN(nn.Module):
    """Simple SCNN module: message passing along four directions.
    This is a minimal, non-optimized variant for demonstration.
    """
    def __init__(self, channels: int, iterations: int = 4):
        super().__init__()
        self.iter = iterations
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Simplified directional passing by repeated 3x3 convs
        for _ in range(self.iter):
            x = x + self.conv(x)
        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 128, out_ch: int = 1):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_ch, mid_ch//2, 3, padding=1), nn.BatchNorm2d(mid_ch//2), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_ch//2, out_ch, 1)
        )

    def forward(self, x):
        return self.decode(x)


class SCNNSegModel(nn.Module):
    def __init__(self, backbone: str = 'resnet18', out_ch: int = 1):
        super().__init__()
        if backbone == 'resnet18':
            net = resnet18(weights=None)
            enc_ch = 512
        else:
            net = resnet34(weights=None)
            enc_ch = 512
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.scnn = SCNN(enc_ch, iterations=4)
        self.head = SegmentationHead(enc_ch)

    def forward(self, x):
        input_size = x.shape[2:]  # H, W
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.scnn(x)
        logits = self.head(x)
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return {"seg_logits": logits}
