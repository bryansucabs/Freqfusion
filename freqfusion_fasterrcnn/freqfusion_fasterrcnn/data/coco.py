"""COCO dataset helpers for Faster R-CNN training."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torchvision.datasets import CocoDetection

from .transforms import Compose, get_eval_transforms, get_train_transforms


class CocoDetectionBoundingBox(CocoDetection):
    """Converts COCO annotations into the format expected by torchvision detectors."""

    def __init__(
        self,
        root: str,
        ann_file: str,
        transforms: Compose | None = None,
    ) -> None:
        super().__init__(root, ann_file)
        self._transforms = transforms

    def __getitem__(self, index: int):
        img, annotations = super().__getitem__(index)
        image_id = self.ids[index]

        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for obj in annotations:
            if obj.get("iscrowd", 0) == 1:
                continue

            x_min, y_min, width, height = obj["bbox"]
            x_max = x_min + width
            y_max = y_min + height

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(int(obj["category_id"]))
            areas.append(float(obj.get("area", width * height)))
            iscrowd.append(0)

        target: Dict[str, Any] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }

        if target["boxes"].numel() == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def build_datasets(
    dataset_root: str,
    train_images: str,
    val_images: str,
    train_annotations: str,
    val_annotations: str,
) -> Tuple[CocoDetectionBoundingBox, CocoDetectionBoundingBox]:
    """Creates COCO train/validation datasets with the default transforms."""

    dataset_root = str(Path(dataset_root).resolve())
    train_set = CocoDetectionBoundingBox(
        root=str(Path(dataset_root, train_images)),
        ann_file=str(Path(dataset_root, train_annotations)),
        transforms=get_train_transforms(),
    )
    val_set = CocoDetectionBoundingBox(
        root=str(Path(dataset_root, val_images)),
        ann_file=str(Path(dataset_root, val_annotations)),
        transforms=get_eval_transforms(),
    )

    return train_set, val_set


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    return tuple(zip(*batch))
