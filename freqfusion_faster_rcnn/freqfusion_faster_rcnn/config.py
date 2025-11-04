"""Default hyper-parameters for the FreqFusion Faster R-CNN experiment."""

from __future__ import annotations

DEFAULT_CONFIG = {
    "num_classes": 91,  # COCO has 80 classes + background index 0 + 10 reserved (following torchvision convention)
    "min_im_size": 600,
    "max_im_size": 1000,
    "backbone_out_channels": 256,
    "rpn_bg_threshold": 0.3,
    "rpn_fg_threshold": 0.7,
    "rpn_batch_size": 256,
    "rpn_pos_fraction": 0.5,
    "rpn_nms_threshold": 0.7,
    "rpn_train_topk": 2000,
    "rpn_test_topk": 1000,
    "rpn_train_prenms_topk": 2000,
    "rpn_test_prenms_topk": 1000,
    "roi_batch_size": 128,
    "roi_pos_fraction": 0.25,
    "roi_iou_threshold": 0.5,
    "roi_low_bg_iou": 0.0,
    "roi_nms_threshold": 0.5,
    "roi_topk_detections": 100,
    "roi_score_threshold": 0.05,
    "roi_pool_size": 7,
    "fc_inner_dim": 1024,
    "scales": [32, 64, 128],
    "aspect_ratios": [0.5, 1.0, 2.0],
}
