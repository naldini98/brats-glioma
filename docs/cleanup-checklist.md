# Checklist: corrida limpia del pipeline

Usá esta lista antes de **re-ejecutar Stage 0** (o todo el pipeline) para evitar mezclar artefactos de una arquitectura antigua (p. ej. `output/stage_2/nnunet_raw/`) con la actual.

**Raíz de datos:** `{project_root}` en `config.yaml` (carpeta que contiene `Task01_BrainTumour/` y `output/`).  
**Código:** carpeta del repo `brats-glioma-seg/`.

---

## 1. Configuración

- [ ] Revisar **`config.yaml`** en la raíz del repo (único archivo de config).
- [ ] En `stage_2` **no** debe haber `nnunet_raw` ni `val` en ratios (solo `train` / `test`).
- [ ] `stage_3.nnunet_raw`: `output/stage_3/nnunet_raw`.
- [ ] `stage_4.nnunet_raw`: `output/stage_3/nnunet_raw` (mismo raw que Stage 3).
- [ ] `nnunet.dataset_id`: `101`, `dataset_name`: `Dataset101_BraTS`.

---

## 2. Materia prima (no borrar)

- [ ] `{project_root}/Task01_BrainTumour/imagesTr/` y `labelsTr/` presentes.
- [ ] `dataset.json` en Task01.
- [ ] **No** se usa `imagesTs` del Decathlon (sin labels públicas).

---

## 3. Limpieza de `output/` (recomendado — corrida full)

### Opción A — Borrar todo `output/` (más simple)

Desde PowerShell (ajustá `project_root`):

```powershell
$root = "C:\Users\franc\FRANCO\PROYECTOS\GLIOMAS CEREBRALES"
Remove-Item -Recurse -Force "$root\output" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path "$root\output" | Out-Null
```

### Opción B — Borrar solo stages (conservar otros archivos en `output/`)

```powershell
$root = "C:\Users\franc\FRANCO\PROYECTOS\GLIOMAS CEREBRALES"
$out = Join-Path $root "output"
foreach ($s in @("stage_0","stage_1","stage_2","stage_3","stage_4")) {
  Remove-Item -Recurse -Force (Join-Path $out $s) -ErrorAction SilentlyContinue
}
```

### Legacy — rutas de pipeline antiguo (eliminar si existen)

- [ ] `{project_root}/output/stage_2/nnunet_raw/` (nnU-Net vivía en Stage 2).
- [ ] `{project_root}/output/stage_3/nnunet_preprocessed/` y `nnunet_results/` si eran **train** en la numeración vieja (ahora train va en `stage_4/`).
- [ ] `{project_root}/output/stage_2/splits/val/` (partición val eliminada; solo train + test).

---

## 4. Entorno y tests

Desde `brats-glioma-seg/`:

```bash
conda activate brats-glioma
pip install -r requirements.txt
python -m pytest tests/unit/ -q
```

- [ ] Tests en verde (30 tests).

---

## 5. Orden de ejecución

Desde `brats-glioma-seg/`:

```bash
python main.py --stage stage_0 --yes
python tools/audit_stage_0.py

python main.py --stage stage_1 --yes
python tools/audit_stage_1.py

python main.py --stage stage_2 --yes
python tools/audit_stage_2.py

python main.py --stage stage_3 --yes
python tools/audit_stage_3.py

python main.py --stage stage_4 --yes
python tools/audit_stage_4.py
```

Cadena hasta Stage 3 (sin entrenar): `python main.py --stage all --yes`

---

## 6. Verificación post-Stage 2

- [ ] `output/stage_2/dataset_split.json` con claves `train` y `test` (no `val`).
- [ ] `output/stage_2/splits/train/imagesTr/` + `labelsTr/` con mismos casos que `manifest.train`.
- [ ] `output/stage_2/splits/test/imagesTs/` + `labelsTs/` con casos `manifest.test`.
- [ ] `output/stage_2/test_ground_truth/*.nii.gz` = un archivo por caso test.

---

## 7. Verificación post-Stage 3

- [ ] `output/stage_3/nnunet_raw/Dataset101_BraTS/imagesTr/` + `labelsTr/` (~85 % casos).
- [ ] `output/stage_3/nnunet_raw/Dataset101_BraTS/imagesTs/` (~15 % casos, sin labels en raw).
- [ ] `dataset.json` → `numTraining` = número de casos train.

---

## 8. Stage 4 (entrenamiento)

- [ ] GPU CUDA disponible (recomendado).
- [ ] `nnUNetv2_plan_and_preprocess` y `nnUNetv2_train` en PATH (`pip install nnunetv2`).
- [ ] Salidas en `output/stage_4/nnunet_preprocessed/` y `nnunet_results/`.
- [ ] Evaluación ciega Dice/HD95: **pendiente de implementación** (`src/eval/blind_eval.py`).

---

## Referencia rápida de carpetas (actual)

```
{project_root}/output/
  stage_0/   # NIfTI nativos por modalidad
  stage_1/   # RAS + 1mm + z-score
  stage_2/   # split train/test + test_ground_truth
  stage_3/   # nnunet_raw/Dataset101_BraTS
  stage_4/   # preprocessed, results, test_predictions
```

Ver detalle: [pipeline.md](pipeline.md).
