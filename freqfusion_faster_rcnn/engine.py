import math
import sys
from typing import Iterable

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_convert

import utils


def _get_iou_types(model):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if hasattr(model_without_ddp, "roi_heads"):
        if hasattr(model_without_ddp.roi_heads, "mask_predictor") and model_without_ddp.roi_heads.mask_predictor is not None:
            iou_types.append("segm")
    return iou_types


def train_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, data_loader: Iterable,
                    device: torch.device, epoch: int, print_freq: int, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        if not math.isfinite(losses_reduced.item()):
            print(f"Loss is {losses_reduced.item()}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def evaluate(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco_gt = COCO(str(data_loader.dataset.ann_file))
    coco_results = []
    iou_types = _get_iou_types(model)

    with torch.inference_mode():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]
            for output, target in zip(outputs, targets):
                boxes = output.get("boxes")
                scores = output.get("scores")
                labels = output.get("labels")
                if boxes is None or boxes.numel() == 0:
                    continue
                boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="xywh")
                image_id = int(target["image_id"].item())
                for box, score, label in zip(boxes, scores, labels):
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": int(label.item()),
                        "bbox": box.tolist(),
                        "score": float(score.item()),
                    })

    torch.set_num_threads(n_threads)

    if not coco_results:
        print("No hay resultados para evaluar.")
        return {}

    coco_dt = coco_gt.loadRes(coco_results)

    stats = {}
    for iou_type in iou_types:
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats[iou_type] = {"AP": coco_eval.stats[0], "AP50": coco_eval.stats[1], "AP75": coco_eval.stats[2]}

    return stats
