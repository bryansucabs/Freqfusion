"""COCO dataset helpers for the FreqFusion Faster R-CNN training script."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

from .transforms import coco_transform


class CocoDetectionDataset(CocoDetection):
    def __init__(self, root: Path, annotation: Path, transforms: Callable = coco_transform):
        super().__init__(root=str(root), annFile=str(annotation), transform=None, target_transform=None)
        self._transforms = transforms

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        boxes = []
        labels = []
        for obj in target:
            bbox = obj["bbox"]
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(obj["category_id"])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        target_dict = {"boxes": boxes_tensor, "labels": labels_tensor}
        if self._transforms is not None:
            image, target_dict = self._transforms(image, target_dict)
        return image, target_dict


def collate(batch: Iterable[Tuple]):
    image, target = batch[0]
    return image, target


def make_dataloader(root: Path, annotation: Path, shuffle: bool = True) -> DataLoader:
    dataset = CocoDetectionDataset(root=root, annotation=annotation)
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, collate_fn=collate)
