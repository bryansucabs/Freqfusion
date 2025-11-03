"""Training and evaluation loops for the detector."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

from .metrics import MetricLogger


def _move_to_device(data: Iterable, device: torch.device):
    return [item.to(device) for item in data]


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader,
    device: torch.device,
    epoch: int,
    print_freq: int = 50,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    warmup_iters: int = 0,
    warmup_factor: float = 1.0 / 1000,
) -> Dict[str, float]:
    model.train()
    metric_logger = MetricLogger(delimiter=" | ")

    base_lrs = [group["lr"] for group in optimizer.param_groups]
    warmup_active = epoch == 0 and warmup_iters > 0
    if warmup_active:
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            param_group["lr"] = base_lr * warmup_factor

    for step, (images, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header=f"Epoch: [{epoch}]")
    ):
        if warmup_active and step <= warmup_iters:
            warmup_progress = float(step) / float(max(1, warmup_iters))
            factor = warmup_factor + warmup_progress * (1.0 - warmup_factor)
            for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
                param_group["lr"] = base_lr * factor

        images = _move_to_device(images, device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()
        if not torch.isfinite(torch.tensor(loss_value)):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        metric_logger.update(loss=loss_value, **{f"loss_{k}": v for k, v in loss_dict.items()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
        param_group["lr"] = base_lr

    return {name: meter.global_avg for name, meter in metric_logger.meters.items()}


def evaluate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    print_freq: int = 50,
    output_json: Optional[Path] = None,
    max_detections: int = 100,
) -> List[Dict]:
    model.eval()
    metric_logger = MetricLogger(delimiter=" | ")
    results: List[Dict] = []

    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, print_freq, header="Test:"):
            images = _move_to_device(images, device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                prediction = {k: v.to("cpu") for k, v in output.items()}
                scores = prediction.get("scores")
                if scores is not None and scores.numel() > max_detections:
                    keep = torch.topk(scores, max_detections).indices
                    prediction = {k: v[keep] for k, v in prediction.items() if isinstance(v, torch.Tensor)}
                results.append(
                    {
                        "image_id": int(target["image_id"].item()),
                        "prediction": prediction,
                    }
                )

    if output_json is not None:
        serializable = []
        for item in results:
            boxes = item["prediction"].get("boxes")
            scores = item["prediction"].get("scores")
            labels = item["prediction"].get("labels")
            if boxes is None or scores is None or labels is None:
                continue
            for box, score, label in zip(boxes, scores, labels):
                x_min, y_min, x_max, y_max = box.tolist()
                serializable.append(
                    {
                        "image_id": item["image_id"],
                        "category_id": int(label.item()),
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "score": float(score.item()),
                    }
                )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(serializable, f)

    return results
