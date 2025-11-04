"""ResNet50 backbone with the FreqFusion neck."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision

from .freqfusion import FreqFusion


class ResNetFreqFusionBackbone(nn.Module):
    """ResNet50 feature extractor fused by FreqFusion into a single map."""

    def __init__(self, out_channels: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.fuse_high = FreqFusion(hr_channels=1024, lr_channels=2048, out_channels=out_channels)
        self.fuse_mid = FreqFusion(hr_channels=512, lr_channels=out_channels, out_channels=out_channels)
        self._out_channels = out_channels

        for param in self.layer1.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p4 = self.fuse_high(c4, c5)
        fused = self.fuse_mid(c3, p4)
        return fused, (c3, c4, c5)

    @property
    def out_channels(self) -> int:
        return self._out_channels


def build_backbone(device: torch.device, out_channels: int = 256) -> ResNetFreqFusionBackbone:
    backbone = ResNetFreqFusionBackbone(out_channels=out_channels)
    return backbone.to(device)
