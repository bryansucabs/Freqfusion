# FreqFusion + Faster R-CNN (COCO)

Esta carpeta contiene una implementación autocontenida para entrenar y evaluar un detector Faster R-CNN con un módulo de fusión FreqFusion sobre un backbone ResNet-FPN utilizando el conjunto de datos COCO 2017. El objetivo es que puedas ejecutar el proyecto directamente desde Visual Studio Code dentro de un entorno virtual de Python.

## Estructura

```
freqfusion_fasterrcnn/
├── configs/
│   └── default.yaml
├── dataset/
│   └── COCO/
│       ├── annotations_trainval2017/
│       ├── train2017/
│       └── val2017/
├── freqfusion_fasterrcnn/
│   ├── data/
│   │   ├── __init__.py
│   │   └── coco.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py
│   │   └── detector.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── metrics.py
│   └── __init__.py
├── requirements.txt
└── train.py
```

Coloca las carpetas descargadas de COCO exactamente dentro de `dataset/COCO/` (por ejemplo `train2017/`, `val2017/` y `annotations_trainval2017/annotations/instances_train2017.json`).

## Preparación del entorno (VS Code)

1. **Crear entorno virtual**

   Abre la carpeta `freqfusion_fasterrcnn` en VS Code. Luego abre una terminal integrada y ejecuta:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows usa: .venv\Scripts\activate
   ```

   VS Code detectará el entorno virtual `.venv`. Selecciónalo como intérprete de Python si es necesario (`Ctrl+Shift+P` → `Python: Select Interpreter`).

2. **Instalar dependencias**

   Con el entorno activado, instala los requisitos:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > **Nota:** Si deseas usar pesos pre-entrenados de ImageNet para ResNet necesitarás acceso a internet para que `torchvision` descargue los archivos la primera vez. Si no dispones de conexión, deja `use_pretrained_backbone: false` en la configuración.

3. **Preparar datos de COCO**

   Copia manualmente los archivos del dataset COCO 2017 en `dataset/COCO/` siguiendo esta estructura:

   ```
   dataset/COCO/
   ├── annotations_trainval2017/
   │   └── annotations/
   │       ├── instances_train2017.json
   │       └── instances_val2017.json
   ├── train2017/
   │   └── *.jpg
   └── val2017/
       └── *.jpg
   ```

   Actualiza las rutas en `configs/default.yaml` si tus archivos `.json` tienen nombres distintos.

## Ejecución

### Entrenamiento

```bash
python train.py \
  --config configs/default.yaml \
  --output-dir runs/experiment_1
```

- `--config`: Ruta al archivo YAML con la configuración.
- `--output-dir`: Carpeta donde se guardarán los checkpoints, registros y métricas.

El script generará automáticamente la carpeta de salida, guardará un checkpoint (`model_latest.pth`) y un registro en formato JSON (`metrics_epoch_*.json`).

### Evaluación rápida

Para obtener pérdidas en el conjunto de validación sin actualizar pesos:

```bash
python train.py --config configs/default.yaml --evaluate --resume runs/experiment_1/model_latest.pth
```

## Configuración

`configs/default.yaml` expone parámetros esenciales:

- `dataset.root`: Carpeta base donde está `dataset/COCO`.
- `dataset.train` y `dataset.val`: Subrutas para imágenes de entrenamiento y validación.
- `dataset.annotations.train` y `dataset.annotations.val`: Archivos JSON de anotaciones.
- `training.batch_size`, `training.epochs`, `training.learning_rate`.
- `model.use_pretrained_backbone`: Habilita los pesos de ImageNet para ResNet.
- `model.freqfusion.scale_factor`: Factor de escala usado para adaptar la resolución baja a la alta dentro del módulo FreqFusion.

Ajusta estos valores según tus recursos y necesidades.

## Notas adicionales

- El módulo FreqFusion se importa desde `FreqFusion.py` (ubicado en la raíz del repositorio). No es necesario moverlo.
- `requirements.txt` incluye `pycocotools`; en Windows puedes necesitar `Microsoft C++ Build Tools`. Si prefieres evitar compilar, utiliza WSL o un entorno Linux.
- Para reanudar un entrenamiento interrumpido añade `--resume ruta/al/checkpoint.pth`.

¡Listo! Con esto deberías poder entrenar Faster R-CNN con FreqFusion sobre COCO dentro de VS Code.
