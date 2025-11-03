from typing import Optional

import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from .freqfusion_backbone import FreqFusionBackbone


def create_freqfusion_faster_rcnn(num_classes: int,
                                  pretrained_backbone: bool = True,
                                  trainable_backbone_layers: int = 3,
                                  min_size: int = 800,
                                  max_size: int = 1333,
                                  image_mean: Optional[list[float]] = None,
                                  image_std: Optional[list[float]] = None) -> nn.Module:
    backbone = FreqFusionBackbone(
        backbone_name="resnet50",
        pretrained=pretrained_backbone,
        trainable_layers=trainable_backbone_layers,
    )

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=min_size,
        max_size=max_size,
        image_mean=image_mean,
        image_std=image_std,
    )

    return model
