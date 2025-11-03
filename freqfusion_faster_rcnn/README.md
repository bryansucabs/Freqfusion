# FreqFusion + Faster R-CNN

Esta carpeta contiene una implementación lista para ejecutar de Faster R-CNN con un backbone ResNet-50 y un módulo de fusión frecuencial **FreqFusion** integrado en la pirámide de características. El proyecto está preparado para entrenar y evaluar con el conjunto de datos COCO 2017.

## Estructura esperada del proyecto

```
freqfusion_faster_rcnn/
├── README.md
├── requirements.txt
├── train.py
├── engine.py
├── utils.py
├── transforms.py
├── datasets/
│   └── coco.py
└── models/
    ├── __init__.py
    ├── freqfusion_module.py
    ├── freqfusion_backbone.py
    └── freqfusion_faster_rcnn.py
```

Coloque el dataset en `freqfusion_faster_rcnn/dataset/COCO` con la misma estructura oficial de COCO 2017:

```
dataset/COCO/
├── annotations_trainval2017/
│   ├── annotations/
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   └── ...
├── train2017/
│   └── *.jpg
└── val2017/
    └── *.jpg
```

> **Nota:** Solo se necesitan los archivos de anotaciones `instances_train2017.json` e `instances_val2017.json` además de las carpetas de imágenes `train2017` y `val2017`.

## Requisitos

1. Instale [Visual Studio Code](https://code.visualstudio.com/) y abra la carpeta `freqfusion_faster_rcnn` como workspace.
2. Instale la extensión oficial **Python** de VS Code.
3. Cree y active un entorno virtual dentro de la carpeta del proyecto:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\\Scripts\\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Configure VS Code para usar el intérprete del entorno virtual (`.venv`).

## Ejecución del entrenamiento

Asegúrese de que el entorno virtual esté activo y ejecute:

```bash
python train.py \
    --data-root dataset/COCO \
    --train-ann annotations_trainval2017/annotations/instances_train2017.json \
    --val-ann annotations_trainval2017/annotations/instances_val2017.json \
    --epochs 12 \
    --batch-size 2 \
    --num-workers 4
```

### Parámetros principales

- `--data-root`: Ruta a la carpeta `dataset/COCO` (por defecto `dataset/COCO`).
- `--train-ann`: Ruta relativa al archivo de anotaciones de entrenamiento.
- `--val-ann`: Ruta relativa al archivo de anotaciones de validación.
- `--epochs`: Número de épocas de entrenamiento.
- `--batch-size`: Tamaño del lote por dispositivo.
- `--lr`: Tasa de aprendizaje inicial (por defecto `0.02`).
- `--resume`: Ruta a un checkpoint `.pth` para reanudar.
- `--output-dir`: Carpeta donde se guardarán checkpoints y registros (por defecto `runs`).

## Resultados y evaluación

El script `train.py` guarda checkpoints periódicos en la carpeta de salida (`runs/fecha-hora`). Si se proporciona la bandera `--evaluate`, se ejecutará únicamente la evaluación sobre el conjunto de validación utilizando las métricas de COCO.

## Consejos adicionales

- Para reducir el consumo de memoria, ajuste `--batch-size` y `--num-workers` según su GPU/CPU.
- Puede habilitar entrenamiento mixto en precisión flotante (`--amp`) si su GPU lo soporta.
- La integración de FreqFusion se encuentra en `models/freqfusion_backbone.py` y `models/freqfusion_module.py`. Estos módulos amplifican la información de alta y baja frecuencia en la pirámide de características utilizada por Faster R-CNN.

## Licencias

Este proyecto utiliza PyTorch y torchvision bajo sus respectivas licencias. COCO se distribuye bajo la licencia Creative Commons Attribution 4.0.
