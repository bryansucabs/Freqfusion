"""Light-weight PyTorch implementation of the FreqFusion neck."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowPassFilter(nn.Module):
    """Adaptive low-pass filter implemented with depth-wise convolutions."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.filter = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        nn.init.constant_(self.filter.weight, 1.0 / (kernel_size * kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.filter(x)


class HighPassFilter(nn.Module):
    """Adaptive high-pass filter implemented with depth-wise convolutions."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.filter = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.filter.weight, a=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return self.filter(x - base)


class FreqFusion(nn.Module):
    """Fuse high- and low-resolution features following the FreqFusion paper."""

    def __init__(
        self,
        hr_channels: int,
        lr_channels: int,
        out_channels: int,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode

        self.hr_proj = nn.Conv2d(hr_channels, out_channels, kernel_size=1)
        self.lr_proj = nn.Conv2d(lr_channels, out_channels, kernel_size=1)

        self.low_pass = LowPassFilter(out_channels, kernel_size=5)
        self.high_pass = HighPassFilter(out_channels, kernel_size=3)

        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.output = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        hr_feat = self.hr_proj(hr)
        lr_feat = self.lr_proj(lr)

        low_freq = self.low_pass(lr_feat)
        low_freq = F.interpolate(low_freq, size=hr_feat.shape[-2:], mode=self.upsample_mode, align_corners=False)

        high_freq = self.high_pass(hr_feat)

        gate = self.gate(torch.cat([high_freq, low_freq], dim=1))
        fused = gate * low_freq + (1.0 - gate) * high_freq
        return self.output(torch.cat([fused, hr_feat], dim=1))
