# FreqFusion + Faster R-CNN (ResNet50, COCO)

This folder contains a self-contained implementation of the FreqFusion neck plugged into a Faster R-CNN detector with a ResNet-50 backbone. The goal is to provide a runnable training / inference pipeline over the COCO dataset using only PyTorch and TorchVision so you can focus on extending the FreqFusion idea.

## Folder layout

```
freqfusion_faster_rcnn/
├── freqfusion_faster_rcnn/        # Python package with the model, data helpers and config
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── faster_rcnn.py
│   └── models/
│       ├── __init__.py
│       ├── backbone.py
│       ├── freqfusion.py
│       ├── roi_head.py
│       ├── rpn.py
│       └── utils.py
├── requirements.txt               # Minimal dependencies (Torch, TorchVision, COCO tools)
├── scripts/
│   ├── infer.py                   # Run inference on a folder of images
│   └── train.py                   # Train on COCO
└── README.md
```

## 1. Prepare your Python environment in VS Code

1. Create and activate a virtual environment inside the repository root:
   ```bash
   cd /path/to/Freqfusion
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\\Scripts\\activate
   ```
2. Install the dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r freqfusion_faster_rcnn/requirements.txt
   ```
3. In VS Code press <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd>, pick **Python: Select Interpreter** and choose the `.venv` you just created.

## 2. Download COCO into `dataset/COCO`

The scripts expect the following directory layout directly under the repository root:

```
dataset/
└── COCO/
    ├── annotations/
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    ├── train2017/
    └── val2017/
```

You can download the files manually from [cocodataset.org](https://cocodataset.org/#download) or use the helper snippet below (Linux/macOS):

```bash
mkdir -p dataset/COCO
cd dataset/COCO
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

> 💡 Make sure `annotations/instances_train2017.json` and `annotations/instances_val2017.json` exist; the scripts will fail otherwise.

## 3. Training

The `scripts/train.py` file contains a simple single-image training loop for Faster R-CNN. It logs the running losses every epoch and stores checkpoints in `checkpoints/`.

```bash
python scripts/train.py \
  --data-root dataset \
  --epochs 12 \
  --device cuda
```

Key flags:

- `--data-root`: folder containing the `COCO` directory shown above.
- `--epochs`: number of epochs (12 corresponds to the 1x schedule).
- `--device`: `cuda` or `cpu`.
- `--no-pretrained-backbone`: skip loading ImageNet weights for the ResNet backbone (useful for offline setups).
- `--checkpoint`: (optional) resume from a saved `.pt` file.

Checkpoints include the model weights, optimizer, epoch and configuration so you can resume training.

## 4. Inference

Use `scripts/infer.py` to run detection on arbitrary images. It loads a checkpoint, performs inference, and stores visualisations in `outputs/`.

```bash
python scripts/infer.py \
  --checkpoint checkpoints/latest.pt \
  --image-folder demo_images \
  --device cuda \
  --score-threshold 0.5
```

Additional options:

- `--output-folder`: where to store the annotated images (defaults to `outputs/`).
- `--no-pretrained-backbone`: instantiate the backbone without ImageNet weights before loading the checkpoint (use this if you trained with that flag).

The script reads all `.jpg` / `.png` files inside `--image-folder`, applies the detector and saves images with predicted boxes and labels. Adjust `--score-threshold` to control how confident the predictions must be.

## 5. Extending / experimenting

- The model definition lives in `freqfusion_faster_rcnn/faster_rcnn.py` and mirrors the reference Faster R-CNN pipeline.
- The custom FreqFusion neck is implemented in `models/freqfusion.py` and is wired into the ResNet backbone in `models/backbone.py`.
- Hyper-parameters are grouped in `config.py`. You can override any key by editing the file or passing a small dict when calling `build_freqfusion_faster_rcnn`.

## 6. Troubleshooting

- TorchVision's COCO dataset wrapper needs the `pycocotools` package. If you are on Windows and pip fails to install it, use `pip install pycocotools-windows` instead.
- If you run out of GPU memory, decrease the image resolution by lowering `min_im_size` and `max_im_size` in `config.py`.
- The current training script processes one image at a time. To scale to larger batch sizes you can extend the collate function in `data/dataset.py` and adjust the model code accordingly.

With this setup the repository now contains all Python sources required to build and run the FreqFusion + Faster R-CNN model on COCO using the ResNet backbone.
