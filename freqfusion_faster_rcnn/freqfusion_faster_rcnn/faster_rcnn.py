"""Top level factory for the FreqFusion Faster R-CNN model."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .config import DEFAULT_CONFIG
from .models.backbone import ResNetFreqFusionBackbone
from .models.rpn import RegionProposalNetwork
from .models.roi_head import ROIHead
from .models.utils import transform_boxes_to_original_size


class FasterRCNN(nn.Module):
    def __init__(self, model_config: Dict, num_classes: int, pretrained_backbone: bool = True) -> None:
        super().__init__()
        self.model_config = model_config
        self.backbone = ResNetFreqFusionBackbone(
            out_channels=model_config['backbone_out_channels'],
            pretrained=pretrained_backbone,
        )
        self.rpn = RegionProposalNetwork(
            model_config['backbone_out_channels'],
            scales=model_config['scales'],
            aspect_ratios=model_config['aspect_ratios'],
            model_config=model_config,
        )
        self.roi_head = ROIHead(model_config, num_classes, in_channels=model_config['backbone_out_channels'])
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = model_config['min_im_size']
        self.max_size = model_config['max_im_size']

    def normalize_resize_image_and_boxes(self, image, bboxes):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        image = (image - mean[:, None, None]) / std[:, None, None]

        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(float(self.min_size) / min_size, float(self.max_size) / max_size)
        scale_factor = scale.item()

        image = torch.nn.functional.interpolate(
            image,
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=True,
            align_corners=False,
        )

        if bboxes is not None:
            ratios = [
                torch.tensor(s, dtype=torch.float32, device=bboxes.device)
                / torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                for s, s_orig in zip(image.shape[-2:], (h, w))
            ]
            ratio_height, ratio_width = ratios
            xmin, ymin, xmax, ymax = bboxes.unbind(2)
            xmin = xmin * ratio_width
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height
            ymax = ymax * ratio_height
            bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=2)
        return image, bboxes

    def forward(self, image, target=None):
        old_shape = image.shape[-2:]
        if self.training:
            image, bboxes = self.normalize_resize_image_and_boxes(image, target['bboxes'])
            target['bboxes'] = bboxes
        else:
            image, _ = self.normalize_resize_image_and_boxes(image, None)

        feat, _ = self.backbone(image)
        rpn_output = self.rpn(image, feat, target)
        proposals = rpn_output['proposals']
        frcnn_output = self.roi_head(feat, proposals, image.shape[-2:], target)
        if not self.training:
            frcnn_output['boxes'] = transform_boxes_to_original_size(
                frcnn_output['boxes'],
                image.shape[-2:],
                old_shape,
            )
        return rpn_output, frcnn_output


def build_freqfusion_faster_rcnn(config: Dict | None = None, *, pretrained_backbone: bool = True) -> FasterRCNN:
    model_config = DEFAULT_CONFIG.copy()
    if config is not None:
        model_config.update(config)
    model = FasterRCNN(
        model_config=model_config,
        num_classes=model_config['num_classes'],
        pretrained_backbone=pretrained_backbone,
    )
    return model
