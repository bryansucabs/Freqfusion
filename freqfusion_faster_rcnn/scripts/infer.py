"""Inference script for FreqFusion + Faster R-CNN."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import torch
from torchvision.transforms import functional as F

from freqfusion_faster_rcnn import build_freqfusion_faster_rcnn

COCO_CATEGORIES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with FreqFusion + Faster R-CNN")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the checkpoint file")
    parser.add_argument("--image-folder", type=Path, required=True, help="Folder containing images")
    parser.add_argument("--output-folder", type=Path, default=Path("outputs"))
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Initialise the backbone without ImageNet weights before loading the checkpoint",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, model: torch.nn.Module) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])


def load_image(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(image)
    return tensor.unsqueeze(0)


def draw_predictions(image: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    image = image.mul(255).byte().cpu().numpy().transpose(1, 2, 0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{COCO_CATEGORIES[label]}: {score:.2f}"
        cv2.putText(image, text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    model = build_freqfusion_faster_rcnn(pretrained_backbone=not args.no_pretrained_backbone)
    load_checkpoint(args.checkpoint, model)
    model.to(device)
    model.eval()

    args.output_folder.mkdir(parents=True, exist_ok=True)

    image_paths: List[Path] = [
        *args.image_folder.glob("*.jpg"),
        *args.image_folder.glob("*.jpeg"),
        *args.image_folder.glob("*.png"),
    ]
    for path in image_paths:
        image_tensor = load_image(path).to(device)
        with torch.no_grad():
            _, frcnn_output = model(image_tensor, target=None)
        boxes = frcnn_output["boxes"].cpu()
        labels = frcnn_output["labels"].cpu()
        scores = frcnn_output["scores"].cpu()

        visual = draw_predictions(image_tensor[0], boxes, labels, scores, args.score_threshold)
        output_path = args.output_folder / path.name
        cv2.imwrite(str(output_path), visual)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
