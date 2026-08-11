# Registro de Experimentos

Última actualización: 7 de agosto de 2026  
Proyecto: brats-glioma-seg (Medical Decathlon Task01)  
Raíz datos: ver `project_root` en `config.yaml`

## Bitácora

| Fecha | Hito | Detalle |
|-------|------|---------|
| 2026-08-07 | Pipeline renumerado | S2 split 85/15, S3 nnU-Net raw, S4 plan/train. Docs + cleanup checklist. |
| 2026-07-28 | nnU-Net empaquetado | Dataset101_BraTS, clases {0,1,2,3}. |
| 2026-07-23 | Proyecto reubicado | Ruta sin acentos. |
| 2026-07-22 | Stage 1 | RAS, 1mm, clip + z-score. |
| 2026-07-17 | Stage 0 | 484 casos train Decathlon, espacio nativo. |
| - | Corrida full post-refactor | Pendiente (ver cleanup-checklist.md) |
| - | Stage 4 eval Dice/HD95 | Pendiente implementación |

> **Nota:** Corridas anteriores con Stage 2 = conversión nnU-Net en `output/stage_2/nnunet_raw/` **no son compatibles** con la numeración actual. Regenerar desde Stage 0 o limpiar `output/` según [cleanup-checklist.md](cleanup-checklist.md).

## Experimento 001 — Baseline (post-refactor, plantilla)

Completar tras la próxima corrida end-to-end.

| Campo | Valor |
|-------|-------|
| Dataset | Task01_BrainTumour (~484 train con labels) |
| Split Stage 2 | train 85% / test 15%, seed 42 |
| Normalización | clip [0.5, 99.5] + z-score por modalidad/caso |
| Spacing | 1.0 mm isotrópico, RAS |
| Clases | {0, 1, 2, 3} |
| nnU-Net | Dataset101_BraTS, 3d_fullres |
| Canales | 0000=t1, 0001=t1ce, 0002=t2, 0003=flair |
| CV entrenamiento | 5-fold nnU-Net sobre `imagesTr` (Stage 4) |

### Comandos

```bash
python main.py --stage stage_0 --yes
python main.py --stage stage_1 --yes
python main.py --stage stage_2 --yes
python main.py --stage stage_3 --yes
python main.py --stage stage_4 --yes
```

### Rutas clave (actual)

| Stage | Artefacto principal |
|-------|---------------------|
| 2 | `output/stage_2/splits/`, `dataset_split.json` |
| 3 | `output/stage_3/nnunet_raw/Dataset101_BraTS/` |
| 4 | `output/stage_4/nnunet_results/` |
| Test GT | `output/stage_2/test_ground_truth/` |

## Experimento 000 — Histórico (pre-refactor)

Referencia de corrida antigua (Stage 2 = convert nnU-Net, todo en imagesTr sin split test).

| Campo | Valor |
|-------|-------|
| Stage 0–1 | 484/484 OK |
| Stage 2 (viejo) | 484 → `output/stage_2/nnunet_raw/` |
| Stage 3 (viejo) | plan/train en `output/stage_3/` |

## Notas

- MRI: no escala [0,1] ni ventana HU.
- seg siempre nearest en resample.
- Sin remap a BraTS clásico {0,1,2,4}.
- Stage 4 requiere GPU para 3d_fullres en la práctica.
- Test Decathlon (`imagesTs`) no se usa; test = split interno 15% con labels.
