"""Region Proposal Network used by the FreqFusion Faster R-CNN model."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torchvision

from .utils import (
    apply_regression_pred_to_anchors_or_proposals,
    boxes_to_transformation_targets,
    clamp_boxes_to_image_boundary,
    get_iou,
    sample_positive_negative,
)


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, scales, aspect_ratios, model_config):
        super().__init__()
        self.scales = scales
        self.low_iou_threshold = model_config['rpn_bg_threshold']
        self.high_iou_threshold = model_config['rpn_fg_threshold']
        self.rpn_nms_threshold = model_config['rpn_nms_threshold']
        self.rpn_batch_size = model_config['rpn_batch_size']
        self.rpn_pos_count = int(model_config['rpn_pos_fraction'] * self.rpn_batch_size)
        self.rpn_train_topk = model_config['rpn_train_topk']
        self.rpn_test_topk = model_config['rpn_test_topk']
        self.rpn_train_prenms_topk = model_config['rpn_train_prenms_topk']
        self.rpn_test_prenms_topk = model_config['rpn_test_prenms_topk']
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)

        self.rpn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_layer = nn.Conv2d(in_channels, self.num_anchors, kernel_size=1, stride=1)
        self.bbox_reg_layer = nn.Conv2d(in_channels, self.num_anchors * 4, kernel_size=1, stride=1)

        for layer in [self.rpn_conv, self.cls_layer, self.bbox_reg_layer]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def generate_anchors(self, image, feat):
        grid_h, grid_w = feat.shape[-2:]
        image_h, image_w = image.shape[-2:]

        stride_h = torch.tensor(max(image_h // grid_h, 1), dtype=torch.int64, device=feat.device)
        stride_w = torch.tensor(max(image_w // grid_w, 1), dtype=torch.int64, device=feat.device)

        scales = torch.as_tensor(self.scales, dtype=feat.dtype, device=feat.device)
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=feat.dtype, device=feat.device)

        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        base_anchors = base_anchors.round()

        shifts_x = torch.arange(0, grid_w, dtype=torch.int32, device=feat.device) * stride_w
        shifts_y = torch.arange(0, grid_h, dtype=torch.int32, device=feat.device) * stride_h
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)
        shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1)

        anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4))
        anchors = anchors.reshape(-1, 4)
        return anchors

    def assign_targets_to_anchors(self, anchors, gt_boxes):
        if gt_boxes.numel() == 0:
            device = anchors.device
            labels = torch.zeros((anchors.size(0),), dtype=torch.float32, device=device)
            matched_gt_boxes = torch.zeros_like(anchors)
            return labels, matched_gt_boxes

        iou_matrix = get_iou(gt_boxes, anchors)
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        best_match_gt_idx_pre_thresholding = best_match_gt_idx.clone()

        below_low_threshold = best_match_iou < self.low_iou_threshold
        between_thresholds = (best_match_iou >= self.low_iou_threshold) & (best_match_iou < self.high_iou_threshold)
        best_match_gt_idx[below_low_threshold] = -1
        best_match_gt_idx[between_thresholds] = -2

        best_anchor_iou_for_gt, _ = iou_matrix.max(dim=1)
        gt_pred_pair_with_highest_iou = torch.where(iou_matrix == best_anchor_iou_for_gt[:, None])
        pred_inds_to_update = gt_pred_pair_with_highest_iou[1]
        best_match_gt_idx[pred_inds_to_update] = best_match_gt_idx_pre_thresholding[pred_inds_to_update]

        matched_gt_boxes = gt_boxes[best_match_gt_idx.clamp(min=0)]

        labels = best_match_gt_idx >= 0
        labels = labels.to(dtype=torch.float32)

        background_anchors = best_match_gt_idx == -1
        labels[background_anchors] = 0.0

        ignored_anchors = best_match_gt_idx == -2
        labels[ignored_anchors] = -1.0
        return labels, matched_gt_boxes

    def filter_proposals(self, proposals, cls_scores, image_shape):
        cls_scores = cls_scores.reshape(-1)
        cls_scores = torch.sigmoid(cls_scores)
        prenms_topk = self.rpn_train_prenms_topk if self.training else self.rpn_test_prenms_topk
        _, top_n_idx = cls_scores.topk(min(prenms_topk, len(cls_scores)))

        cls_scores = cls_scores[top_n_idx]
        proposals = proposals[top_n_idx]

        proposals = clamp_boxes_to_image_boundary(proposals, image_shape)

        min_size = 16
        ws, hs = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        proposals = proposals[keep]
        cls_scores = cls_scores[keep]

        if proposals.numel() == 0:
            return proposals, cls_scores

        keep_indices = torchvision.ops.nms(proposals, cls_scores, self.rpn_nms_threshold)
        cls_scores = cls_scores[keep_indices]
        proposals = proposals[keep_indices]

        post_topk = self.rpn_train_topk if self.training else self.rpn_test_topk
        keep_order = cls_scores.sort(descending=True)[1][:post_topk]
        proposals = proposals[keep_order]
        cls_scores = cls_scores[keep_order]
        return proposals, cls_scores

    def forward(self, image, feat, target=None):
        rpn_feat = torch.relu(self.rpn_conv(feat))
        cls_scores = self.cls_layer(rpn_feat)
        box_transform_pred = self.bbox_reg_layer(rpn_feat)

        anchors = self.generate_anchors(image, feat)

        number_of_anchors_per_location = cls_scores.size(1)
        cls_scores = cls_scores.permute(0, 2, 3, 1)
        cls_scores = cls_scores.reshape(-1, 1)

        box_transform_pred = box_transform_pred.view(
            box_transform_pred.size(0),
            number_of_anchors_per_location,
            4,
            rpn_feat.shape[-2],
            rpn_feat.shape[-1],
        )
        box_transform_pred = box_transform_pred.permute(0, 3, 4, 1, 2)
        box_transform_pred = box_transform_pred.reshape(-1, 4)

        proposals = apply_regression_pred_to_anchors_or_proposals(
            box_transform_pred.detach().reshape(-1, 1, 4),
            anchors,
        )
        proposals = proposals.reshape(proposals.size(0), 4)

        proposals, scores = self.filter_proposals(proposals, cls_scores.detach(), image.shape[-2:])
        rpn_output: Dict[str, torch.Tensor] = {
            'proposals': proposals,
            'scores': scores,
        }
        if not self.training or target is None:
            return rpn_output

        labels_for_anchors, matched_gt_boxes_for_anchors = self.assign_targets_to_anchors(
            anchors,
            target['bboxes'][0],
        )
        regression_targets = boxes_to_transformation_targets(matched_gt_boxes_for_anchors, anchors)

        sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
            labels_for_anchors,
            positive_count=self.rpn_pos_count,
            total_count=self.rpn_batch_size,
        )

        sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]
        if sampled_pos_idx_mask.sum() == 0:
            localization_loss = torch.tensor(0.0, device=feat.device)
        else:
            localization_loss = (
                torch.nn.functional.smooth_l1_loss(
                    box_transform_pred[sampled_pos_idx_mask],
                    regression_targets[sampled_pos_idx_mask],
                    beta=1 / 9,
                    reduction="sum",
                )
                / (sampled_idxs.numel())
            )

        cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            cls_scores[sampled_idxs].flatten(),
            labels_for_anchors[sampled_idxs].flatten(),
        )

        rpn_output['rpn_classification_loss'] = cls_loss
        rpn_output['rpn_localization_loss'] = localization_loss
        return rpn_output
