import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import engine
import utils
from datasets import CocoDetection
from models import create_freqfusion_faster_rcnn


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento de Faster R-CNN con FreqFusion sobre COCO")
    parser.add_argument("--data-root", default="dataset/COCO", help="Ruta a la carpeta base del dataset COCO")
    parser.add_argument("--train-ann", default="annotations_trainval2017/annotations/instances_train2017.json",
                        help="Ruta relativa al archivo de anotaciones de entrenamiento")
    parser.add_argument("--val-ann", default="annotations_trainval2017/annotations/instances_val2017.json",
                        help="Ruta relativa al archivo de anotaciones de validación")
    parser.add_argument("--epochs", default=12, type=int, help="Número de épocas")
    parser.add_argument("--batch-size", default=2, type=int, help="Tamaño del lote por dispositivo")
    parser.add_argument("--num-workers", default=4, type=int, help="Número de workers para DataLoader")
    parser.add_argument("--lr", default=0.02, type=float, help="Tasa de aprendizaje inicial")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum del optimizador")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="Regularización L2")
    parser.add_argument("--lr-steps", default=[8, 11], nargs="+", type=int, help="Épocas para disminuir la LR")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="Factor de decaimiento de LR")
    parser.add_argument("--output-dir", default="runs", help="Carpeta de salida para checkpoints")
    parser.add_argument("--resume", default="", help="Ruta a un checkpoint para reanudar")
    parser.add_argument("--num-classes", default=91, type=int, help="Número de clases (incluye fondo)")
    parser.add_argument("--trainable-backbone-layers", default=3, type=int, choices=range(5),
                        help="Número de etapas entrenables del backbone ResNet")
    parser.add_argument("--no-pretrained-backbone", action="store_true", help="No cargar pesos ImageNet en el backbone")
    parser.add_argument("--amp", action="store_true", help="Usar entrenamiento con precisión mixta")
    parser.add_argument("--evaluate", action="store_true", help="Solo evaluar el modelo")
    return parser.parse_args()


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Usando dispositivo: {device}")

    torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root)
    dataset_train = CocoDetection(str(data_root), args.train_ann, train=True)
    dataset_val = CocoDetection(str(data_root), args.val_ann, train=False)

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
    )

    model = create_freqfusion_faster_rcnn(
        num_classes=args.num_classes,
        pretrained_backbone=not args.no_pretrained_backbone,
        trainable_backbone_layers=args.trainable_backbone_layers,
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Checkpoint cargado desde {args.resume}, reanudando en la época {start_epoch}")

    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluate:
        stats = engine.evaluate(model, data_loader_val, device=device)
        if stats:
            print(json.dumps(stats, indent=2))
        return

    for epoch in range(start_epoch, args.epochs):
        engine.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=20, scaler=scaler)
        lr_scheduler.step()

        checkpoint_path = output_dir / f"model_epoch_{epoch}.pth"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }, checkpoint_path)
        print(f"Checkpoint guardado en {checkpoint_path}")

        stats = engine.evaluate(model, data_loader_val, device=device)
        if stats:
            stats_path = output_dir / f"metrics_epoch_{epoch}.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            print(f"Resultados de evaluación guardados en {stats_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
