"""Utility functions shared across the Faster R-CNN components."""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor


def get_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute the pairwise IoU for two sets of boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)), device=boxes1.device if boxes1.numel() else boxes2.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union = area1[:, None] + area2 - intersection_area
    return intersection_area / union.clamp(min=1e-6)


def boxes_to_transformation_targets(
    ground_truth_boxes: Tensor, anchors_or_proposals: Tensor
) -> Tensor:
    widths = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    heights = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * widths
    center_y = anchors_or_proposals[:, 1] + 0.5 * heights

    gt_widths = ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]
    gt_heights = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
    gt_center_x = ground_truth_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = ground_truth_boxes[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_center_x - center_x) / widths.clamp(min=1e-6)
    targets_dy = (gt_center_y - center_y) / heights.clamp(min=1e-6)
    targets_dw = torch.log(gt_widths / widths.clamp(min=1e-6))
    targets_dh = torch.log(gt_heights / heights.clamp(min=1e-6))
    return torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)


def apply_regression_pred_to_anchors_or_proposals(
    box_transform_pred: Tensor, anchors_or_proposals: Tensor
) -> Tensor:
    box_transform_pred = box_transform_pred.reshape(box_transform_pred.size(0), -1, 4)

    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * w
    center_y = anchors_or_proposals[:, 1] + 0.5 * h

    dx = box_transform_pred[..., 0]
    dy = box_transform_pred[..., 1]
    dw = box_transform_pred[..., 2].clamp(max=math.log(1000.0 / 16))
    dh = box_transform_pred[..., 3].clamp(max=math.log(1000.0 / 16))

    pred_center_x = dx * w[:, None] + center_x[:, None]
    pred_center_y = dy * h[:, None] + center_y[:, None]
    pred_w = torch.exp(dw) * w[:, None]
    pred_h = torch.exp(dh) * h[:, None]

    pred_box_x1 = pred_center_x - 0.5 * pred_w
    pred_box_y1 = pred_center_y - 0.5 * pred_h
    pred_box_x2 = pred_center_x + 0.5 * pred_w
    pred_box_y2 = pred_center_y + 0.5 * pred_h

    pred_boxes = torch.stack(
        (pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2),
        dim=2,
    )
    return pred_boxes


def sample_positive_negative(labels: Tensor, positive_count: int, total_count: int) -> Tuple[Tensor, Tensor]:
    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels == 0)[0]

    num_pos = min(positive.numel(), positive_count)
    num_neg = min(negative.numel(), total_count - num_pos)

    if positive.numel() == 0:
        perm_positive_idxs = positive
    else:
        perm_positive_idxs = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    if negative.numel() == 0:
        perm_negative_idxs = negative
    else:
        perm_negative_idxs = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idxs = positive[perm_positive_idxs]
    neg_idxs = negative[perm_negative_idxs]

    sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool)

    sampled_pos_idx_mask[pos_idxs] = True
    sampled_neg_idx_mask[neg_idxs] = True
    return sampled_neg_idx_mask, sampled_pos_idx_mask


def clamp_boxes_to_image_boundary(boxes: Tensor, image_shape: Tuple[int, int]) -> Tensor:
    boxes_x1 = boxes[..., 0].clamp(min=0, max=image_shape[1])
    boxes_y1 = boxes[..., 1].clamp(min=0, max=image_shape[0])
    boxes_x2 = boxes[..., 2].clamp(min=0, max=image_shape[1])
    boxes_y2 = boxes[..., 3].clamp(min=0, max=image_shape[0])

    boxes = torch.stack((boxes_x1, boxes_y1, boxes_x2, boxes_y2), dim=-1)
    return boxes


def transform_boxes_to_original_size(boxes: Tensor, new_size: Tuple[int, int], original_size: Tuple[int, int]) -> Tensor:
    if boxes.numel() == 0:
        return boxes
    ratios = [
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
