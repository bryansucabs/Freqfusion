"""Datasets y transformaciones para COCO."""

from .coco import build_datasets, collate_fn

__all__ = ["build_datasets", "collate_fn"]
