# Registro de Experimentos

Última actualización: 28 de julio de 2026  
Proyecto: brats-glioma-seg (Medical Decathlon Task01)  
Raíz: `C:/Users/franc/FRANCO/PROYECTOS/GLIOMAS CEREBRALES`

## Bitácora

| Fecha | Hito | Detalle |
|-------|------|---------|
| 2026-07-28 | Stage 3 implementado | plan_and_preprocess + train 3d_fullres fold 0. |
| 2026-07-28 | Stage 2 completo | 484/484 SUCCESS_CONVERTED. Dataset101_BraTS listo. |
| 2026-07-28 | Stage 2 implementado | Convert a nnU-Net v2 Dataset101_BraTS. Clases {0,1,2,3}. |
| 2026-07-23 | Proyecto reubicado | Ruta sin acentos. output/ vacío, regenerar Stage 0. |
| 2026-07-22 | Stage 1 implementado | RAS, resample 1mm, clip percentil + z-score por modalidad/caso. |
| 2026-07-17 | Stage 0 completo | 484/484 SUCCESS_GEOMETRIC_APPROVED. |
| 2026-07-17 | Stage 0 Decathlon | Split 4D, clases {0,1,2,3}, espacio nativo. |
| - | Stage 3 Train | Implementado; corrida pendiente |
| - | Stage 4 Eval | Pendiente |

## Experimento 000 - Baseline

| Campo | Valor |
|-------|-------|
| Dataset | Task01_BrainTumour (484 train) |
| Stage 0 | 484 SUCCESS_GEOMETRIC_APPROVED |
| Stage 1 | 484 SUCCESS_PREPROCESSED |
| Stage 2 | 484 SUCCESS_CONVERTED |
| Normalización | clip [0.5, 99.5] + z-score por modalidad/caso |
| Spacing | 1.0 mm isotrópico |
| Orientación | RAS |
| Clases | {0, 1, 2, 3} |
| nnU-Net dataset | Dataset101_BraTS (id 101) |
| Canales nnU-Net | 0000=t1, 0001=t1ce, 0002=t2, 0003=flair |

### Stage 1

| Métrica | Valor |
|---------|-------|
| Casos procesados | 484 |
| SUCCESS_PREPROCESSED | 484 |
| FAILED | 0 |

Comando: `python main.py --stage stage_1 --yes`

### Stage 2

| Métrica | Valor |
|---------|-------|
| Casos convertidos | 484 |
| SUCCESS_CONVERTED | 484 |
| FAILED | 0 |
| dataset.json numTraining | 484 |
| Auditoría | 484 OK, 0 FAILED |

Comando: `python main.py --stage stage_2 --yes`  
Auditoría: `python tools/audit_stage_2.py`  
Dataset: `data/nnunet_raw/Dataset101_BraTS/`

### Stage 3 (completar tras corrida)

| Métrica | Valor |
|---------|-------|
| plan_and_preprocess | - |
| train fold 0 | - |
| device | - |
| checkpoint_final | - |

Comando: `python main.py --stage stage_3 --yes`  
Solo plan: `python scripts/train_nnunet.py --plan-only --yes`  
Auditoría: `python tools/audit_stage_3.py`

## Notas

- MRI: no se usa escalado [0,1] ni ventana HU (a diferencia del pipeline TAC nódulos).
- seg siempre nearest en resample.
- Stage 2 no remapea clases a BraTS clásico {0,1,2,4}.
- Stage 2 no altera geometría/intensidades: solo copia + dataset.json.
- Stage 3 requiere GPU (CUDA) para ser práctico; CPU es solo depuración.
- Splits por case_id (BRATS_XXX); folds 5-CV los genera nnU-Net en Stage 3.
