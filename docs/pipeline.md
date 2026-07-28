# Pipeline BraTS Glioma Segmentation

Última actualización: 28 de julio de 2026  
Proyecto: brats-glioma-seg (Medical Decathlon Task01_BrainTumour)  
Raíz del proyecto: `C:/Users/franc/FRANCO/PROYECTOS/GLIOMAS CEREBRALES`  
Estado: Stages 0–3 implementados. Stage 4 (eval) pendiente.

## Flujo general

```
Task01_BrainTumour/  (materia prima Decathlon)
    |
    v  Stage 0: volúmenes NIfTI mellizos (espacio nativo)
output/stage_0/
    |
    v  Stage 1: RAS + 1mm + clip/z-score
output/stage_1/
    |
    v  Stage 2: Convert nnU-Net v2
data/nnunet_raw/Dataset101_BraTS/
    |
    v  Stage 3: Plan/preprocess + Train nnU-Net
data/nnunet_preprocessed/ + data/nnunet_results/
    |
    v  Stage 4: Eval Dice (pendiente)
```

## Stage 0: Generación de Volúmenes y Máscaras Base

Comando: `python main.py --stage stage_0 --yes`

- Entrada: Task01_BrainTumour (4D + máscara 3D)
- Split 4 canales → t1, t1ce, t2, flair + seg
- Espacio NATIVO (sin RAS, resample ni normalización)
- Clases seg: {0, 1, 2, 3} (Decathlon)
- Gate: SUCCESS_GEOMETRIC_APPROVED
- Auditoría: `python tools/audit_stage_0.py`

## Stage 1: Preprocesamiento Espacial e Intensidad

Comando: `python main.py --stage stage_1 --yes`

Entrada: solo casos SUCCESS_GEOMETRIC_APPROVED de `output/stage_0/`.

### Pasos por paciente

| Paso | Qué hace | Por qué |
|------|----------|---------|
| 1. Reorientación RAS | Alinea ejes anatómicos | Convención estándar 3D / nnU-Net |
| 2. Resample 1 mm | Spacing isotrópico | Homogeneizar resolución entre casos |
| 3. Interpolación | MRI: linear, seg: nearest | Intensidades vs etiquetas discretas |
| 4. Normalización | Clip [p0.5, p99.5] + z-score por modalidad/caso | MRI no tiene escala absoluta (no HU) |
| 5. Gate | verify_stage_1_output | Mellizos + spacing 1mm + seg {0,1,2,3} |

### Normalización (sin escalado [0,1])

A diferencia del pipeline de nódulos pulmonares (TAC):

- No hay ventana HU fija ni escala [0,1].
- Por cada paciente y cada modalidad por separado:
  1. Máscara cerebral desde FLAIR (voxels > 0)
  2. Clip a percentiles 0.5 y 99.5 (robusto a outliers)
  3. Z-score: (x - μ) / σ sobre voxeles cerebrales

Resultado: media ~0, desvío ~1 por modalidad. Compatible con nnU-Net.

### Salidas Stage 1

| Artefacto | Ruta |
|-----------|------|
| Volúmenes normalizados | output/stage_1/01_nifti_volumes/volume_{case_id}/ |
| Máscaras resampleadas | output/stage_1/02_nifti_masks/volume_{case_id}/ |
| Log | output/stage_1/preprocessing_log.csv |
| Auditoría | output/stage_1/reporte_auditoria_stage_1.md |

Estado exitoso: SUCCESS_PREPROCESSED

Auditoría: `python tools/audit_stage_1.py`

### Configuración (config.yaml → stage_1)

```yaml
stage_1:
  orientation: RAS
  target_spacing: [1.0, 1.0, 1.0]
  reference_modality: t1
  normalization:
    method: percentile_clip_zscore
    percentile_low: 0.5
    percentile_high: 99.5
    mask_source: flair
```

### Módulos

- `src/data/spatial_preprocess.py` — RAS + resample
- `src/data/intensity_norm.py` — clip + z-score
- `src/data/stage_1.py` — orquestación y gate
- `tools/audit_stage_1.py` — auditoría post-hoc

## Stage 2: Conversión a formato nnU-Net v2

Comando: `python main.py --stage stage_2 --yes`  
Alias: `python scripts/convert_to_nnunet.py --yes`

Entrada: solo casos SUCCESS_PREPROCESSED de `output/stage_1/`.

### Qué hace

| Paso | Qué hace | Por qué |
|------|----------|---------|
| 1. Filtrar casos | Lee preprocessing_log Stage 1 | Solo casos aprobados |
| 2. Copiar canales | `volume_*-{mod}.nii.gz` → `{case}_000X.nii.gz` | Naming oficial nnU-Net |
| 3. Copiar label | `volume_*-seg.nii.gz` → `{case}.nii.gz` | Una máscara por caso |
| 4. dataset.json | channel_names + labels + numTraining | Metadatos nnU-Net v2 |
| 5. Gate | verify_stage_2_output | 4 canales + label, geometría melliza, clases {0,1,2,3} |

**No altera** geometría ni intensidades: solo empaqueta. Sin remap a BraTS clásico `{0,1,2,4}`.

### Naming nnU-Net (orden de canales)

| Canal | Modalidad | Archivo |
|-------|-----------|---------|
| `_0000` | t1 | `BRATS_001_0000.nii.gz` |
| `_0001` | t1ce | `BRATS_001_0001.nii.gz` |
| `_0002` | t2 | `BRATS_001_0002.nii.gz` |
| `_0003` | flair | `BRATS_001_0003.nii.gz` |
| label | seg | `labelsTr/BRATS_001.nii.gz` |

### Salidas Stage 2

| Artefacto | Ruta |
|-----------|------|
| Dataset nnU-Net | `data/nnunet_raw/Dataset101_BraTS/` |
| Imágenes | `.../imagesTr/` |
| Etiquetas | `.../labelsTr/` |
| Metadatos | `.../dataset.json` |
| Log | `output/stage_2/conversion_log.csv` |
| Auditoría | `output/stage_2/reporte_auditoria_stage_2.md` |

Estado exitoso: SUCCESS_CONVERTED

Auditoría: `python tools/audit_stage_2.py`

### Configuración (config.yaml → stage_2 / nnunet)

```yaml
stage_2:
  nnunet_raw: "data/nnunet_raw"
  modalities: [t1, t1ce, t2, flair]
  label_file: seg

nnunet:
  dataset_id: 101
  dataset_name: Dataset101_BraTS
  configuration: 3d_fullres
  fold: 0
```

### Módulos

- `src/data/nnunet_convert.py` — naming, dataset.json, copia NIfTI
- `src/data/stage_2.py` — orquestación y gate
- `scripts/convert_to_nnunet.py` — wrapper CLI
- `tools/audit_stage_2.py` — auditoría post-hoc

### Siguiente paso

Stage 3: `python main.py --stage stage_3 --yes`

## Stage 3: Plan/preprocess + Train (nnU-Net v2)

Comando: `python main.py --stage stage_3 --yes`  
Aliases: `python main.py --stage train --yes` · `python scripts/train_nnunet.py --yes`

Entrada: `data/nnunet_raw/Dataset101_BraTS/` (Stage 2 completo).

### Qué hace (2 pasos)

| Paso | Comando nnU-Net | Qué hace |
|------|-----------------|----------|
| 1. Plan + preprocess | `nnUNetv2_plan_and_preprocess -d 101` | Fingerprint del dataset, planes (patch/batch/spacing), preprocesado interno nnU-Net |
| 2. Train | `nnUNetv2_train 101 3d_fullres 0` | Entrena U-Net 3D full-res en fold 0 (5-fold CV); ~1000 epochs por defecto |

Stage 3 **no** reescribe Stage 1/2: nnU-Net aplica su propio preprocesado encima del raw empaquetado.

### Variables de entorno (seteadas por el stage)

| Variable | Ruta (bajo project_root) |
|----------|--------------------------|
| `nnUNet_raw` | `data/nnunet_raw` |
| `nnUNet_preprocessed` | `data/nnunet_preprocessed` |
| `nnUNet_results` | `data/nnunet_results` |

### Salidas Stage 3

| Artefacto | Ruta |
|-----------|------|
| Planes + datos preprocesados | `data/nnunet_preprocessed/Dataset101_BraTS/` |
| Checkpoints | `data/nnunet_results/Dataset101_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/` |
| Log | `output/stage_3/training_log.csv` |
| Auditoría | `output/stage_3/reporte_auditoria_stage_3.md` |

Estados: `SUCCESS_PLANNED` (paso 1) · `SUCCESS_TRAINED` (paso 2) · `FAILED`

Auditoría: `python tools/audit_stage_3.py`

### Configuración clave (`stage_3`)

```yaml
stage_3:
  run_plan_and_preprocess: true
  run_train: true
  configuration: 3d_fullres
  folds: [0]          # o all / [0,1,2,3,4]
  device: auto        # cuda | cpu | mps
  np: 2
  continue_training: false
```

Solo plan: `python scripts/train_nnunet.py --plan-only --yes`  
Solo train (si ya hay planes): `python scripts/train_nnunet.py --train-only --yes`

### Módulos

- `src/models/nnunet_env.py` — env vars, device, CLI
- `src/models/stage_3.py` — orquestación y gates
- `scripts/train_nnunet.py` — wrapper CLI
- `tools/audit_stage_3.py` — auditoría post-hoc

### Notas prácticas

- **GPU recomendada.** Sin CUDA el train en CPU es impráctico para BraTS 3D.
- Default nnU-Net: ~1000 epochs/fold; fold 0 basta para baseline.
- Para retomar: `continue_training: true` (flag `--c`).

## Clases y subregiones (Decathlon)

| Valor | Clase |
|-------|-------|
| 0 | background |
| 1 | edema |
| 2 | non-enhancing |
| 3 | enhancing |

- WT: [1, 2, 3]
- TC: [2, 3]
- ET: [3]

## Estructura en disco

```
C:/Users/franc/FRANCO/PROYECTOS/GLIOMAS CEREBRALES/
├── Task01_BrainTumour/          # materia prima Decathlon
├── brats-glioma-seg/            # código + config.yaml
├── data/
│   ├── nnunet_raw/
│   │   └── Dataset101_BraTS/    # Stage 2
│   ├── nnunet_preprocessed/     # Stage 3
│   └── nnunet_results/          # Stage 3
└── output/
    ├── stage_0/
    ├── stage_1/
    ├── stage_2/
    └── stage_3/
```

`stage_2`/`stage_3` paths se resuelven bajo `project_root`.

## Historial

| Fecha | Cambio |
|-------|--------|
| 2026-07-28 | Stage 3: plan/preprocess + train nnU-Net v2 (Dataset101, 3d_fullres). |
| 2026-07-28 | Stage 2: conversión a nnU-Net v2 (Dataset101_BraTS), clases Decathlon. |
| 2026-07-23 | Proyecto movido a ruta sin acentos. write_nifti con verificación read-after-write. |
| 2026-07-22 | Stage 1: RAS + 1mm + clip/z-score por modalidad/caso. |
| 2026-07-17 | Stage 0: Decathlon 4D → NIfTI mellizos nativos. |
