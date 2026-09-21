"""LiteBallNet architecture matching the handed-off demonstration checkpoints."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=32, min_channels=8):
        super().__init__()
        hidden = max(min_channels, channels // reduction)
        self.shared = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish(inplace=True)
        self.attn_h = nn.Conv2d(hidden, channels, 1, bias=True)
        self.attn_w = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x):
        height = x.shape[2]
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        shared = self.act(self.bn(self.shared(torch.cat([x_h, x_w], dim=2))))
        feature_h, feature_w = torch.split(shared, [height, x.shape[3]], dim=2)
        feature_w = feature_w.transpose(2, 3)
        return x * torch.sigmoid(self.attn_h(feature_h)) * torch.sigmoid(self.attn_w(feature_w))


class LiteBallNet(nn.Module):
    """Depthwise-separable U-Net with bottleneck coordinate attention."""

    def __init__(self, base_channels=24, use_ca=True, ca_reduction=32, ca_min_channels=8):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.stem = DSConvBlock(9, c1)
        self.enc1 = DSConvBlock(c1, c1)
        self.down1 = DSConvBlock(c1, c2, stride=2)
        self.enc2 = DSConvBlock(c2, c2)
        self.down2 = DSConvBlock(c2, c3, stride=2)
        self.bottleneck1 = DSConvBlock(c3, c3)
        self.bottleneck2 = DSConvBlock(c3, c3)
        self.ca = CoordinateAttention(c3, ca_reduction, ca_min_channels) if use_ca else nn.Identity()
        self.dec2 = DSConvBlock(c3 + c2, c2)
        self.dec1 = DSConvBlock(c2 + c1, c1)
        self.head = nn.Conv2d(c1, 1, 1, bias=True)

    def forward(self, x):
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        x = self.bottleneck2(self.bottleneck1(self.down2(e2)))
        x = self.ca(x)
        x = F.interpolate(x, size=e2.shape[-2:], mode='nearest')
        x = self.dec2(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode='nearest')
        x = self.dec1(torch.cat([x, e1], dim=1))
        return torch.sigmoid(self.head(x))
