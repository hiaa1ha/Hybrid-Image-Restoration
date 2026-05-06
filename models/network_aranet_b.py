"""
network_aranet_b.py

ARANet-B: Attention Residual Architecture Network (variant B)
for blind image denoising.

Architecture overview:
  - Head conv:        maps input (n_channels + 1 noise-map) -> nf feature maps
  - Encoder path:     3 x (GroupedResidualBlock stack + stride-2 conv)
  - Bottleneck:       Spatial Attention Module (SAM) + Frequency Attention Module (FAM)
  - Decoder path:     3 x (GroupedResidualBlock stack + ConvTranspose2d x2) with skip connections
  - Tail conv:        maps nf feature maps -> output channels

Usage:
    from models.network_aranet_b import UNetRes
    model = UNetRes(in_nc=4, out_nc=3, nf=64, nb=4)
"""

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self._bias = self.pointwise.bias

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

    @property
    def weight(self):
        return self.pointwise.weight

    @property
    def bias(self):
        return self._bias


class GroupedResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(in_channels, in_channels)
        )

    def forward(self, x):
        return x + self.block(x)


class SAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv(x)


class FAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ffted = torch.fft.fft2(x)
        ffted = torch.abs(torch.fft.fftshift(ffted))
        ffted = torch.fft.ifftshift(ffted)
        freq_att = self.ca(ffted.real)
        return x * freq_att


class UNetRes(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nf=64, nb=2, **kwargs):
        super().__init__()

        self.head = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.down1 = nn.Sequential(
            *[GroupedResidualBlock(nf) for _ in range(nb)],
            nn.Conv2d(nf, nf, 4, 2, 1)
        )
        self.down2 = nn.Sequential(
            *[GroupedResidualBlock(nf) for _ in range(nb)],
            nn.Conv2d(nf, nf, 4, 2, 1)
        )
        self.down3 = nn.Sequential(
            *[GroupedResidualBlock(nf) for _ in range(nb)],
            nn.Conv2d(nf, nf, 4, 2, 1)
        )

        self.sam = SAM(nf)
        self.fam = FAM(nf)

        self.up3 = nn.Sequential(
            *[GroupedResidualBlock(nf) for _ in range(nb)],
            nn.ConvTranspose2d(nf, nf, 4, 2, 1)
        )
        self.up2 = nn.Sequential(
            *[GroupedResidualBlock(nf) for _ in range(nb)],
            nn.ConvTranspose2d(nf, nf, 4, 2, 1)
        )
        self.up1 = nn.Sequential(
            *[GroupedResidualBlock(nf) for _ in range(nb)],
            nn.ConvTranspose2d(nf, nf, 4, 2, 1)
        )

        self.tail = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def forward(self, x):
        x1 = self.head(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x_att = self.sam(x4) + self.fam(x4)

        x = self.up3(x_att)
        x = self.up2(x + x3)
        x = self.up1(x + x2)

        return self.tail(x + x1)


if __name__ == '__main__':
    model = UNetRes()
    x = torch.randn(1, 4, 256, 256)  # 3 channels + noise map
    with torch.no_grad():
        y = model(x)
    print(y.shape)
