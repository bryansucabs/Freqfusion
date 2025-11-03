"""Backbone utilities that insert FreqFusion into a ResNet-FPN extractor."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from FreqFusion import FreqFusion


class FreqFusionFPNBackbone(nn.Module):
    """Wraps a torchvision FPN backbone and fuses extreme pyramid levels."""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        use_pretrained_backbone: bool = False,
        trainable_layers: int = 3,
        freqfusion_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        freqfusion_kwargs = freqfusion_kwargs or {}

        weights = None
        if use_pretrained_backbone:
            try:
                from torchvision.models import ResNet50_Weights, ResNet101_Weights

                if backbone_name == "resnet50":
                    weights = ResNet50_Weights.DEFAULT
                elif backbone_name == "resnet101":
                    weights = ResNet101_Weights.DEFAULT
            except (ImportError, AttributeError):
                # Older torchvision versions use alternative enums or constants.
                weights = "IMAGENET1K_V2"

        self.backbone = resnet_fpn_backbone(
            backbone_name,
            weights=weights,
            trainable_layers=trainable_layers,
        )

        out_channels = getattr(self.backbone, "out_channels", 256)
        self.freqfusion = FreqFusion(
            hr_channels=out_channels,
            lr_channels=out_channels,
            **freqfusion_kwargs,
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        pyramid = self.backbone(x)
        if not isinstance(pyramid, OrderedDict):
            raise TypeError("The wrapped backbone must return an OrderedDict")

        pyramid = OrderedDict(pyramid)
        if "0" in pyramid and "3" in pyramid:
            pyramid["0"] = self.freqfusion(pyramid["0"], pyramid["3"])
        else:
            raise KeyError(
                "FreqFusion expects keys '0' (highest resolution) and '3' (lowest) in the FPN output"
            )
        return pyramid


def build_backbone(
    backbone_name: str,
    use_pretrained_backbone: bool,
    trainable_layers: int,
    freqfusion_kwargs: Optional[Dict[str, Any]] = None,
) -> FreqFusionFPNBackbone:
    """Factory helper used by the detector builder."""

    return FreqFusionFPNBackbone(
        backbone_name=backbone_name,
        use_pretrained_backbone=use_pretrained_backbone,
        trainable_layers=trainable_layers,
        freqfusion_kwargs=freqfusion_kwargs,
    )
