from pathlib import Path
from typing import Any, Dict

from PIL import Image
from pycocotools.coco import COCO
import torch

import transforms as T


class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, root: str, ann_file: str, train: bool = True):
        self.root = Path(root)
        self.ann_file = self.root / ann_file
        if not self.ann_file.exists():
            raise FileNotFoundError(f"No se encontró el archivo de anotaciones: {self.ann_file}")
        self.coco = COCO(str(self.ann_file))
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.train = train
        self.transforms = self._build_transforms(train=train)

    def _build_transforms(self, train: bool):
        if train:
            return T.Compose([
                T.ToTensor(),
                T.RandomHorizontalFlip(0.5),
            ])
        return T.Compose([
            T.ToTensor(),
        ])

    def __getitem__(self, index: int):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]["file_name"]
        img_path = self.root / path
        if not img_path.exists():
            split = "train2017" if self.train else "val2017"
            img_path = self.root / split / path
        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for obj in annotations:
            xmin, ymin, width, height = obj["bbox"]
            xmax = xmin + width
            ymax = ymin + height

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj["category_id"])
            areas.append(obj["area"])
            iscrowd.append(obj.get("iscrowd", 0))

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target: Dict[str, Any] = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor(img_id)
        target["area"] = areas
        target["iscrowd"] = iscrowd

        image, target = self.transforms(image, target)
        return image, target

    def __len__(self) -> int:
        return len(self.ids)
