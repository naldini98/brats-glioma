# brats-glioma-seg

Pipeline de **segmentación multiclase de gliomas cerebrales (BraTS)** con **nnU-Net v2** sobre **Medical Segmentation Decathlon Task01**.

Cuatro modalidades MRI (T1, T1ce, T2, FLAIR), etiquetas Decathlon `{0,1,2,3}` y subregiones clínicas **WT**, **TC** y **ET**.

## Pipeline

```
Task01_BrainTumour/     →  Stage 0  →  output/stage_0/
                         →  Stage 1  →  output/stage_1/
                         →  Stage 2  →  output/stage_2/  (split + test_ground_truth/)
                         →  Stage 3  →  output/stage_3/nnunet_raw/Dataset101_BraTS/
                         →  Stage 4  →  output/stage_4/  (preprocessed, results, eval)
```

| Stage | Qué hace |
|-------|----------|
| **0** | Split 4D Decathlon → NIfTI por modalidad (espacio nativo) |
| **1** | RAS + resample 1 mm + clip/z-score por modalidad/caso |
| **2** | Split train/test: `splits/train` y `splits/test`, imágenes + labels |
| **3** | Empaquetado formato nnU-Net v2 (`Dataset101_BraTS`) |
| **4** | `plan_and_preprocess` + train (+ predict/eval opcional) |

## Requisitos

- Python 3.10+
- Conda env recomendado: `brats-glioma`
- [Medical Segmentation Decathlon Task01_BrainTumour](http://medicaldecathlon.com/) (descarga aparte, **no incluido** en este repo)
- GPU recomendada para Stage 4 (entrenamiento)

```bash
conda create -n brats-glioma python=3.10 -y
conda activate brats-glioma
pip install -r requirements.txt
```

## Configuración

Editá **`config.yaml`** en la raíz del repo (`project_root`, rutas `output/stage_X/`). Es la única fuente de verdad (ver `AGENTS.md`).

Ejemplo de layout local:

```
GLIOMAS CEREBRALES/
├── Task01_BrainTumour/
├── brats-glioma-seg/          ← este repo
├── output/
│   ├── stage_0/ … stage_2/
│   ├── stage_3/nnunet_raw/Dataset101_BraTS/
│   └── stage_4/
│       ├── nnunet_preprocessed/
│       ├── nnunet_results/
│       └── test_predictions/
```

## Uso

Desde `brats-glioma-seg/`:

```bash
python main.py --stage stage_0 --yes
python main.py --stage stage_1 --yes
python main.py --stage stage_2 --yes
python main.py --stage stage_3 --yes
python main.py --stage stage_4 --yes
```

Auditorías:

```bash
python tools/audit_stage_0.py
python tools/audit_stage_1.py
python tools/audit_stage_2.py
python tools/audit_stage_3.py
python tools/audit_stage_4.py
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

Canales nnU-Net (Stage 3): `_0000`=t1, `_0001`=t1ce, `_0002`=t2, `_0003`=flair

## Documentación

- [docs/pipeline.md](docs/pipeline.md) — detalle por stage
- [docs/cleanup-checklist.md](docs/cleanup-checklist.md) — **limpieza de `output/` y corrida desde Stage 0**
- [docs/experiments.md](docs/experiments.md) — bitácora de corridas
- [AGENTS.md](AGENTS.md) — reglas del proyecto

## Licencia / datos

El código de este repositorio es tuyo bajo la licencia que elijas al publicar.  
El dataset **Task01_BrainTumour** tiene licencia **CC-BY-SA 4.0** (Medical Segmentation Decathlon); no se redistribuye aquí.

## Autor

Franco Naldini — [GitHub](https://github.com/naldini98)
