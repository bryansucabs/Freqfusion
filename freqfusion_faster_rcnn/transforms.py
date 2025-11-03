import random
from typing import Iterable

import torch
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms: Iterable):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            width, _ = F.get_image_size(image)
            boxes = target["boxes"]
            boxes = boxes.clone()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target


class ResizeToMultiple:
    def __init__(self, size_divisible=32):
        self.size_divisible = size_divisible

    def __call__(self, image, target):
        _, h, w = image.shape
        new_h = (h + self.size_divisible - 1) // self.size_divisible * self.size_divisible
        new_w = (w + self.size_divisible - 1) // self.size_divisible * self.size_divisible
        if new_h != h or new_w != w:
            image = F.resize(image, [new_h, new_w])
            if "boxes" in target:
                scale_y = new_h / h
                scale_x = new_w / w
                boxes = target["boxes"]
                boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y], dtype=boxes.dtype)
                target["boxes"] = boxes
        return image, target
