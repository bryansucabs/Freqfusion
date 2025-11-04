"""Data transforms for COCO detection."""

from __future__ import annotations

from typing import Dict

import torch
from torchvision.transforms import functional as F


def prepare_target(target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    boxes = target["boxes"].float()
    labels = target["labels"].long()
    return {"bboxes": boxes.unsqueeze(0), "labels": labels.unsqueeze(0)}


def coco_transform(image, target):
    image = F.to_tensor(image)
    boxes = target["boxes"].float()
    labels = target["labels"].long()
    return image, {"bboxes": boxes.unsqueeze(0), "labels": labels.unsqueeze(0)}
