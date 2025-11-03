"""Paquete utilitario para entrenar Faster R-CNN con FreqFusion."""

from .models.detector import build_detector

__all__ = ["build_detector"]
