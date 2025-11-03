"""Detection-friendly transforms adapted from torchvision references."""
from __future__ import annotations

import random
from typing import Dict, Tuple

import torch
from PIL import Image
import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def __call__(self, image: torch.Tensor, target: Dict[str, torch.Tensor]):
        if random.random() < self.probability:
            image = image.flip(-1)
            width = image.shape[-1]
            boxes = target["boxes"]
            boxes = boxes.clone()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target


def get_train_transforms() -> Compose:
    return Compose([ToTensor(), RandomHorizontalFlip(0.5)])


def get_eval_transforms() -> Compose:
    return Compose([ToTensor()])
