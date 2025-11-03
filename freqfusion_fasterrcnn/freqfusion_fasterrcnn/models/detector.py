"""Model factory for the FreqFusion-enhanced Faster R-CNN."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from .backbone import build_backbone


DEFAULT_ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,))
DEFAULT_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(DEFAULT_ANCHOR_SIZES)


def build_detector(
    num_classes: int,
    backbone_name: str = "resnet50",
    use_pretrained_backbone: bool = False,
    trainable_backbone_layers: int = 3,
    freqfusion_kwargs: Optional[Dict[str, Any]] = None,
) -> FasterRCNN:
    """Creates a Faster R-CNN detector instrumented with FreqFusion."""

    backbone = build_backbone(
        backbone_name=backbone_name,
        use_pretrained_backbone=use_pretrained_backbone,
        trainable_layers=trainable_backbone_layers,
        freqfusion_kwargs=freqfusion_kwargs,
    )

    anchor_generator = AnchorGenerator(
        sizes=DEFAULT_ANCHOR_SIZES,
        aspect_ratios=DEFAULT_ASPECT_RATIOS,
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
    )

    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str, strict: bool = True) -> Dict[str, Any]:
    """Loads model parameters from a checkpoint file."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    output_path: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Persists model (and optimizer) state for later reuse."""

    state = {"model": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if metrics is not None:
        state["metrics"] = metrics

    torch.save(state, output_path)
