"""FreqFusion + Faster R-CNN package."""

from .config import DEFAULT_CONFIG
from .faster_rcnn import build_freqfusion_faster_rcnn

__all__ = ["DEFAULT_CONFIG", "build_freqfusion_faster_rcnn"]
