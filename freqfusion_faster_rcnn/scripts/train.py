"""Training script for FreqFusion + Faster R-CNN on COCO."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch import nn, optim
from tqdm import tqdm

from freqfusion_faster_rcnn import DEFAULT_CONFIG, build_freqfusion_faster_rcnn
from freqfusion_faster_rcnn.data.dataset import make_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FreqFusion + Faster R-CNN on COCO")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to the folder containing the COCO directory")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Resume training from a checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Initialise the ResNet backbone without ImageNet weights",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, model: nn.Module, optimizer: optim.Optimizer) -> Dict:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    config: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def train_one_epoch(model: nn.Module, dataloader, optimizer: optim.Optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    losses = {
        "rpn_cls": 0.0,
        "rpn_reg": 0.0,
        "roi_cls": 0.0,
        "roi_reg": 0.0,
    }
    pbar = tqdm(dataloader, desc="train", leave=False)
    for image, target in pbar:
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        optimizer.zero_grad()
        rpn_output, frcnn_output = model(image, target)
        total_loss = (
            rpn_output['rpn_classification_loss']
            + rpn_output['rpn_localization_loss']
            + frcnn_output['frcnn_classification_loss']
            + frcnn_output['frcnn_localization_loss']
        )
        total_loss.backward()
        optimizer.step()

        losses["rpn_cls"] += float(rpn_output['rpn_classification_loss'])
        losses["rpn_reg"] += float(rpn_output['rpn_localization_loss'])
        losses["roi_cls"] += float(frcnn_output['frcnn_classification_loss'])
        losses["roi_reg"] += float(frcnn_output['frcnn_localization_loss'])

        pbar.set_postfix({"loss": total_loss.item()})

    num_batches = len(dataloader)
    return {k: v / num_batches for k, v in losses.items()}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    train_images = args.data_root / "COCO" / "train2017"
    train_annotations = args.data_root / "COCO" / "annotations" / "instances_train2017.json"
    dataloader = make_dataloader(train_images, train_annotations, shuffle=True)

    model = build_freqfusion_faster_rcnn(DEFAULT_CONFIG, pretrained_backbone=not args.no_pretrained_backbone)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    start_epoch = 0
    if args.checkpoint and args.checkpoint.exists():
        checkpoint = load_checkpoint(args.checkpoint, model, optimizer)
        start_epoch = checkpoint.get("epoch", 0) + 1

    for epoch in range(start_epoch, args.epochs):
        avg_losses = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}/{args.epochs}: {avg_losses}")
        save_checkpoint(args.output_dir / f"epoch_{epoch + 1}.pt", model, optimizer, epoch, DEFAULT_CONFIG)
        save_checkpoint(args.output_dir / "latest.pt", model, optimizer, epoch, DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
