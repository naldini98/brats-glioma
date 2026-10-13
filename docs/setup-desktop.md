# Setup — PC escritorio (GPU)

Última actualización: 6 de agosto de 2026

Entorno de referencia para entrenar (Stage 3) con CUDA. Los **datos y artefactos** viven en **X:**; el **entorno Conda** en **Z:**.

## Rutas

| Qué | Ruta |
|-----|------|
| `project_root` (config.yaml) | `X:/FRANCO/PROYECTOS/GLIOMAS CEREBRALES` |
| Dataset Decathlon | `{project_root}/Task01_BrainTumour/` |
| Código | `{project_root}/brats-glioma-seg/` |
| Salidas pipeline | `{project_root}/output/` (stages + nnU-Net) |
| Entorno Conda `brats-glioma` | `Z:\Entornos virtuales\brats-glioma` |

Todas las rutas relativas del `config.yaml` se resuelven bajo `project_root`, no bajo la carpeta del repo.

## 1. Entorno Conda (disco Z)

En PowerShell:

```powershell
conda create --prefix "Z:\Entornos virtuales\brats-glioma" python=3.10 -y
conda activate "Z:\Entornos virtuales\brats-glioma"
cd "X:\FRANCO\PROYECTOS\GLIOMAS CEREBRALES\brats-glioma-seg"
pip install -r requirements.txt
```

En cada sesión nueva:

```powershell
conda activate "Z:\Entornos virtuales\brats-glioma"
cd "X:\FRANCO\PROYECTOS\GLIOMAS CEREBRALES\brats-glioma-seg"
```

### PyTorch + CUDA

`requirements.txt` instala `torch` sin fijar CUDA. Tras el `pip install`, comprobá:

```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

Si sale `False` y tenés GPU NVIDIA, reinstalá PyTorch con el wheel CUDA que corresponda a tu driver desde [pytorch.org](https://pytorch.org) (mismo entorno activo).

## 2. Configuración

1. Copiá `config.example.yaml` → `config.yaml` si no existe (el archivo local está en `.gitignore`).
2. En `config.yaml`, `project_root` debe apuntar a la carpeta que contiene `Task01_BrainTumour/`, `output/` y `brats-glioma-seg/`.

En el escritorio:

```yaml
project_root: "X:/FRANCO/PROYECTOS/GLIOMAS CEREBRALES"
```

## 3. Tests

Con el entorno activo, desde `brats-glioma-seg/`:

```powershell
python -m pytest tests/unit/ -q
```

## 4. Pipeline (orden)

Con `process_test: true` (Stages 0–1) y `convert_test: true` (Stage 2), **cada invocación de stage procesa train y luego test** con el mismo preprocesado.

Usá siempre `--yes` para evitar prompts interactivos.

### Corrida completa desde cero

Borrá (o renombrá) artefactos viejos bajo `project_root`:

- `output/stage_0`, `output/stage_0_test`
- `output/stage_1`, `output/stage_1_test`
- `output/stage_2`, `output/stage_3`
- `output/nnunet_raw`, `output/nnunet_preprocessed`, `output/nnunet_results` (si querés re-entrenar limpio)

Luego, desde `brats-glioma-seg/`:

```powershell
python main.py --stage stage_0 --yes
python main.py --stage stage_1 --yes
python main.py --stage stage_2 --yes
python main.py --stage stage_3 --yes
```

Esperado al final de Stage 2: `Dataset101_BraTS/imagesTr`, `labelsTr`, **`imagesTs`**, y `dataset.json` con `numTraining` + `numTest`.

Auditorías opcionales después de cada stage:

```powershell
python tools/audit_stage_0.py
python tools/audit_stage_1.py
python tools/audit_stage_2.py
python tools/audit_stage_3.py
```

### Prueba rápida (pocos casos)

En `config.yaml`, podés limitar temporalmente:

```yaml
stage_0:
  max_cases: 2
stage_1:
  max_cases: 2
stage_2:
  max_cases: 2
```

Volvé a `max_cases: null` para la corrida completa (484 casos).

## 5. Otra máquina (portátil / sin GPU)

- Mismo repo; otro `project_root` en `config.yaml`.
- Entorno Conda puede ser `conda create -n brats-glioma python=3.10` si preferís nombre corto en la ubicación por defecto de Conda.
- Los artefactos `output/` no se versionan: copiarlos manualmente o regenerar stages en cada máquina.
