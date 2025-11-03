from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from .freqfusion_module import FreqFusion


class FreqFusionBackbone(nn.Module):
    def __init__(self,
                 backbone_name: str = "resnet50",
                 pretrained: bool = True,
                 trainable_layers: int = 3,
                 feature_names: Optional[list[str]] = None):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained and backbone_name == "resnet50" else None
        self.backbone = resnet_fpn_backbone(backbone_name, weights=weights, trainable_layers=trainable_layers)
        self.out_channels = self.backbone.out_channels
        if feature_names is None:
            feature_names = ["0", "1", "2", "3", "pool"]
        self.feature_names = feature_names
        self.freq_modules = nn.ModuleDict()
        for high_name, low_name in zip(self.feature_names[:-1], self.feature_names[1:]):
            self.freq_modules[f"{high_name}->{low_name}"] = FreqFusion(self.out_channels, self.out_channels, scale_factor=2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        fused = OrderedDict()
        prev_name = None
        prev_feat = None
        for name, feat in features.items():
            if prev_feat is not None:
                key = f"{prev_name}->{name}"
                module = self.freq_modules.get(key, None)
                if module is not None:
                    feat = module(prev_feat, feat)
            fused[name] = feat
            prev_name = name
            prev_feat = fused[name]
        return fused

    @property
    def body(self):
        return self.backbone.body

    @property
    def fpn(self):
        return self.backbone.fpn
