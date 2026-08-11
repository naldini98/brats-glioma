# Pipeline BraTS Glioma Segmentation

Última actualización: 7 de agosto de 2026  
Proyecto: brats-glioma-seg (Medical Decathlon Task01_BrainTumour)  
Estado: Stages 0–4 implementados; evaluación ciega (Dice/HD95) en Stage 4 pendiente.

## Flujo general

```
Task01_BrainTumour/  (solo imagesTr + labelsTr; imagesTs Decathlon ignorado)
    |
    v  Stage 0: NIfTI por modalidad (espacio nativo)
output/stage_0/
    |
    v  Stage 1: RAS + 1mm + clip/z-score
output/stage_1/
    |
    v  Stage 2: split train (85%) / test (15%), imágenes + labels emparejados
output/stage_2/splits/ + test_ground_truth/
    |
    v  Stage 3: empaquetado nnU-Net v2
output/stage_3/nnunet_raw/Dataset101_BraTS/
    |
    v  Stage 4: plan_and_preprocess + train (+ predict/eval opcional)
output/stage_4/nnunet_preprocessed/ + nnunet_results/
```

**Validación durante entrenamiento:** nnU-Net hace **5-fold CV** sobre todo `imagesTr`/`labelsTr` (no hay carpeta val en Stage 2).  
**Test ciego:** nunca entra en `labelsTr` del raw nnU-Net; GT en `output/stage_2/test_ground_truth/`.

Corrida limpia: [cleanup-checklist.md](cleanup-checklist.md).

---

## Stage 0: Generación de Volúmenes y Máscaras Base

Comando: `python main.py --stage stage_0 --yes`

- Entrada: Task01 `imagesTr` + `labelsTr` vía `dataset.json`
- Split 4D → t1, t1ce, t2, flair + seg
- Espacio **NATIVO** (sin RAS, resample ni normalización)
- Clases seg: {0, 1, 2, 3} (Decathlon)
- Gate: `SUCCESS_GEOMETRIC_APPROVED`
- Auditoría: `python tools/audit_stage_0.py`

| Salida | Ruta |
|--------|------|
| Volúmenes | `output/stage_0/01_nifti_volumes/volume_{case}/` |
| Máscaras | `output/stage_0/02_nifti_masks/volume_{case}/` |
| Log | `output/stage_0/processing_log.csv` |

---

## Stage 1: Preprocesamiento Espacial e Intensidad

Comando: `python main.py --stage stage_1 --yes` · alias `preprocess`

Entrada: casos `SUCCESS_GEOMETRIC_APPROVED` en log Stage 0.

| Paso | Qué hace |
|------|----------|
| RAS | Convención anatómica |
| Resample 1 mm | Spacing isotrópico |
| MRI linear / seg nearest | |
| Clip [0.5, 99.5] + z-score por modalidad/caso | Sin escala [0,1] |
| Gate | Mellizos + spacing ~1mm + seg {0,1,2,3} |

Estado exitoso: `SUCCESS_PREPROCESSED`  
Auditoría: `python tools/audit_stage_1.py`

Módulos: `src/data/spatial_preprocess.py`, `intensity_norm.py`, `stage_1.py`

---

## Stage 2: Split Manager (train / test)

Comando: `python main.py --stage stage_2 --yes`

Entrada: `output/stage_1/` (casos preprocesados).

| Partición | ~% | Carpetas (bajo `output/stage_2/splits/`) |
|-----------|-----|------------------------------------------|
| **train** | 85 | `train/imagesTr/` + `train/labelsTr/` |
| **test** | 15 | `test/imagesTs/` + `test/labelsTs/` |

Además:

- `output/stage_2/dataset_split.json` — listas `train`, `test`, `counts`, `layout`
- `output/stage_2/test_ground_truth/` — máscaras test en nombre plano (`BRATS_XXX.nii.gz`) para eval Stage 4
- `output/stage_2/processing_log.csv`

Estado exitoso: `SUCCESS_SPLIT`  
Auditoría: `python tools/audit_stage_2.py`

Módulos: `src/data/dataset_split.py`, `src/data/stage_2.py`

Config ejemplo:

```yaml
stage_2:
  splits_dir: "output/stage_2/splits"
  split_manifest: "output/stage_2/dataset_split.json"
  test_ground_truth_dir: "output/stage_2/test_ground_truth"
  ratios:
    train: 0.85
    test: 0.15
  seed: 42
```

---

## Stage 3: Conversión nnU-Net v2

Comando: `python main.py --stage stage_3 --yes` · alias `convert`  
Wrapper: `python scripts/convert_to_nnunet.py --yes`

Entrada: bundles Stage 2 (`splits/train`, `splits/test`).

| Origen | Destino nnU-Net |
|--------|-----------------|
| train (images + labels) | `imagesTr/` + `labelsTr/` |
| test (solo imágenes) | `imagesTs/` |

Salida: `output/stage_3/nnunet_raw/Dataset101_BraTS/` + `dataset.json` (`numTraining` = casos train).

Naming canales: `_0000`=t1, `_0001`=t1ce, `_0002`=t2, `_0003`=flair.

Estado exitoso: `SUCCESS_CONVERTED`  
Auditoría: `python tools/audit_stage_3.py`

Módulos: `src/data/nnunet_convert.py`, `src/data/stage_3.py`

---

## Stage 4: Plan, preprocess y train (nnU-Net v2)

Comando: `python main.py --stage stage_4 --yes` · alias `train`  
Wrapper: `python scripts/train_nnunet.py --yes`

Entrada: `output/stage_3/nnunet_raw/Dataset101_BraTS/`.

| Paso | CLI |
|------|-----|
| Plan + preprocess | `nnUNetv2_plan_and_preprocess -d 101` |
| Train | `nnUNetv2_train 101 3d_fullres {fold}` |

Variables de entorno (seteadas por el stage):

| Variable | Ruta típica |
|----------|-------------|
| `nnUNet_raw` | `output/stage_3/nnunet_raw` |
| `nnUNet_preprocessed` | `output/stage_4/nnunet_preprocessed` |
| `nnUNet_results` | `output/stage_4/nnunet_results` |

Salidas: checkpoints en `.../nnUNetTrainer__nnUNetPlans__3d_fullres/fold_X/`  
Log: `output/stage_4/training_log.csv`  
Auditoría: `python tools/audit_stage_4.py`

Opcional (`config stage_4`): `run_predict`, `run_eval` — eval Dice/HD95 vs `test_ground_truth` **pendiente** (`src/eval/blind_eval.py`).

Solo plan: `python scripts/train_nnunet.py --plan-only --yes`  
Solo train: `python scripts/train_nnunet.py --train-only --yes`

Módulos: `src/models/nnunet_env.py`, `src/models/stage_4.py`

---

## Clases y subregiones (Decathlon)

| Valor | Clase |
|-------|-------|
| 0 | background |
| 1 | edema |
| 2 | non-enhancing |
| 3 | enhancing |

WT: [1,2,3] · TC: [2,3] · ET: [3]

---

## Estructura en disco

```
{project_root}/
├── Task01_BrainTumour/
├── brats-glioma-seg/          # repo + config.yaml
└── output/
    ├── stage_0/
    ├── stage_1/
    ├── stage_2/
    │   ├── splits/train/…
    │   ├── splits/test/…
    │   ├── dataset_split.json
    │   └── test_ground_truth/
    ├── stage_3/
    │   └── nnunet_raw/Dataset101_BraTS/
    └── stage_4/
        ├── nnunet_preprocessed/
        ├── nnunet_results/
        └── test_predictions/
```

Rutas `stage_*` en `config.yaml` se resuelven bajo `project_root`.

---

## Historial

| Fecha | Cambio |
|-------|--------|
| 2026-08-07 | Stage 2 = split train/test; Stage 3 = nnU-Net convert; Stage 4 = plan/train. |
| 2026-08-06 | Salidas bajo `output/` por stage. |
| 2026-07-28 | Conversión nnU-Net v2 Dataset101 (clases Decathlon). |
| 2026-07-22 | Stage 1 RAS + 1mm + z-score. |
| 2026-07-17 | Stage 0 Decathlon 4D → NIfTI nativos. |
