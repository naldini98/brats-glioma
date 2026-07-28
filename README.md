# brats-glioma-seg

Pipeline de **segmentación multiclase de gliomas cerebrales (BraTS)** con **nnU-Net v2** sobre **Medical Segmentation Decathlon Task01**.

Cuatro modalidades MRI (T1, T1ce, T2, FLAIR), etiquetas Decathlon `{0,1,2,3}` y subregiones clínicas **WT**, **TC** y **ET**.

## Pipeline

```
Task01_BrainTumour/     →  Stage 0  →  output/stage_0/
                         →  Stage 1  →  output/stage_1/
                         →  Stage 2  →  data/nnunet_raw/Dataset101_BraTS/
                         →  Stage 3  →  data/nnunet_preprocessed/ + data/nnunet_results/
                         →  Stage 4  →  eval (pendiente)
```

| Stage | Qué hace |
|-------|----------|
| **0** | Split 4D Decathlon → NIfTI por modalidad (espacio nativo) |
| **1** | RAS + resample 1 mm + clip/z-score por modalidad/caso |
| **2** | Empaquetado formato nnU-Net v2 (`Dataset101_BraTS`) |
| **3** | `plan_and_preprocess` + train nnU-Net (`3d_fullres`) |

## Requisitos

- Python 3.10+
- Conda env recomendado: `brats-glioma`
- [Medical Segmentation Decathlon Task01_BrainTumour](http://medicaldecathlon.com/) (descarga aparte, **no incluido** en este repo)
- GPU recomendada para Stage 3 (entrenamiento)

```bash
conda create -n brats-glioma python=3.10 -y
conda activate brats-glioma
pip install -r requirements.txt
```

## Configuración

1. Copiá `config.example.yaml` → `config.yaml` (o editá el existente).
2. Ajustá `project_root` a la carpeta que contiene `Task01_BrainTumour/`, `output/` y `data/`.

Ejemplo de layout local:

```
GLIOMAS CEREBRALES/
├── Task01_BrainTumour/
├── brats-glioma-seg/          ← este repo
├── output/
└── data/
    ├── nnunet_raw/
    ├── nnunet_preprocessed/
    └── nnunet_results/
```

## Uso

Desde `brats-glioma-seg/`:

```bash
python main.py --stage stage_0 --yes
python main.py --stage stage_1 --yes
python main.py --stage stage_2 --yes
python main.py --stage stage_3 --yes
```

Auditorías:

```bash
python tools/audit_stage_0.py
python tools/audit_stage_1.py
python tools/audit_stage_2.py
python tools/audit_stage_3.py
```

Tests:

```bash
python -m pytest tests/unit/ -q
```

## Clases (Decathlon)

| Valor | Clase |
|-------|-------|
| 0 | background |
| 1 | edema |
| 2 | non-enhancing |
| 3 | enhancing |

Subregiones: **WT** `[1,2,3]` · **TC** `[2,3]` · **ET** `[3]`

Canales nnU-Net (Stage 2): `_0000`=t1, `_0001`=t1ce, `_0002`=t2, `_0003`=flair

## Documentación

- [docs/pipeline.md](docs/pipeline.md) — detalle por stage
- [docs/experiments.md](docs/experiments.md) — bitácora de corridas
- [AGENTS.md](AGENTS.md) — reglas del proyecto

## Licencia / datos

El código de este repositorio es tuyo bajo la licencia que elijas al publicar.  
El dataset **Task01_BrainTumour** tiene licencia **CC-BY-SA 4.0** (Medical Segmentation Decathlon); no se redistribuye aquí.

## Autor

Franco Naldini — [GitHub](https://github.com/naldini98)
