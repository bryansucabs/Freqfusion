from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
import yaml

from freqfusion_fasterrcnn.data.coco import build_datasets, collate_fn
from freqfusion_fasterrcnn.models.detector import build_detector, load_checkpoint, save_checkpoint
from freqfusion_fasterrcnn.utils.engine import evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN + FreqFusion on COCO")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Ruta al archivo YAML de configuración")
    parser.add_argument("--output-dir", type=str, default="runs/default", help="Carpeta donde guardar resultados")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint a reanudar")
    parser.add_argument("--evaluate", action="store_true", help="Solo ejecutar evaluación")
    parser.add_argument("--amp", action="store_true", help="Usar mixed precision con CUDA")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone ResNet a utilizar")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_device(preferred: str) -> torch.device:
    preferred = preferred.lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred in {"cpu", "cuda"}:
        return torch.device("cpu") if preferred == "cpu" else torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_dataloaders(cfg: Dict[str, Any]):
    dataset_cfg = cfg["dataset"]
    train_set, val_set = build_datasets(
        dataset_root=dataset_cfg["root"],
        train_images=dataset_cfg["train"],
        val_images=dataset_cfg["val"],
        train_annotations=dataset_cfg["annotations"]["train"],
        val_annotations=dataset_cfg["annotations"]["val"],
    )

    train_loader = DataLoader(
        train_set,
        batch_size=dataset_cfg.get("batch_size", 2),
        shuffle=True,
        num_workers=dataset_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]):
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(
        params,
        lr=cfg["learning_rate"],
        momentum=cfg.get("momentum", 0.9),
        weight_decay=cfg.get("weight_decay", 0.0005),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim,
        step_size=cfg.get("lr_step_size", 8),
        gamma=cfg.get("lr_gamma", 0.1),
    )
    return optim, scheduler


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = ensure_device(cfg.get("training", {}).get("device", "cuda"))
    print(f"Usando dispositivo: {device}")

    train_loader, val_loader = prepare_dataloaders(cfg)

    model_cfg = cfg["model"]
    model = build_detector(
        num_classes=model_cfg.get("num_classes", 91),
        backbone_name=args.backbone,
        use_pretrained_backbone=model_cfg.get("use_pretrained_backbone", False),
        trainable_backbone_layers=model_cfg.get("trainable_backbone_layers", 3),
        freqfusion_kwargs=model_cfg.get("freqfusion", {}),
    )
    model.to(device)

    training_cfg = cfg["training"]
    optimizer, lr_scheduler = build_optimizer(model, training_cfg)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(model, args.resume, strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Reanudando desde la época {start_epoch}")

    if args.evaluate and not args.resume:
        raise ValueError("Debe proporcionar --resume con un checkpoint para evaluar")

    if args.evaluate:
        evaluate(
            model,
            val_loader,
            device=device,
            output_json=output_dir / "predicciones_eval.json",
            max_detections=cfg.get("validation", {}).get("max_detections", 100),
        )
        return

    num_epochs = training_cfg.get("epochs", 12)
    metrics_history = []

    for epoch in range(start_epoch, num_epochs):
        train_stats = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device=device,
            epoch=epoch,
            print_freq=training_cfg.get("print_freq", 50),
            scaler=scaler,
            warmup_iters=training_cfg.get("warmup_iters", 0),
            warmup_factor=training_cfg.get("warmup_factor", 1.0 / 1000),
        )

        lr_scheduler.step()

        metrics_history.append({"epoch": epoch, "train": train_stats})

        if (epoch + 1) % training_cfg.get("checkpoint_interval", 1) == 0:
            predictions = evaluate(
                model,
                val_loader,
                device=device,
                output_json=output_dir / f"predicciones_epoch_{epoch:03d}.json",
                max_detections=cfg.get("validation", {}).get("max_detections", 100),
            )
            metrics_history[-1]["validation"] = {"images": len(predictions)}

            save_checkpoint(
                model,
                optimizer,
                epoch=epoch,
                output_path=str(output_dir / "model_latest.pth"),
                metrics=metrics_history,
            )

            metrics_file = output_dir / f"metrics_epoch_{epoch:03d}.json"
            with metrics_file.open("w", encoding="utf-8") as f:
                json.dump(metrics_history[-1], f, indent=2)

    print("Entrenamiento finalizado")


if __name__ == "__main__":
    main()
