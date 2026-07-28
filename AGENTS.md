# NeuroSeg-BraTS Agent Rules

## Pipeline
stage_0 (Decathlon 4D→NIfTI nativo) -> stage_1 (RAS+1mm+z-score) -> stage_2 (nnU-Net convert) -> stage_3 (plan+train) -> eval

## Reglas Críticas
- Stage 0: ESPACIO NATIVO. Sin RAS, resampling ni normalización.
- Stage 1: RAS + resample 1mm + clip percentil + z-score por modalidad/caso.
- Stage 1: NO escalar a [0,1] (MRI no tiene unidades absolutas como HU).
- Stage 2: solo empaquetado nnU-Net (copia + dataset.json). NO alterar geometría/intensidades.
- Stage 2: naming `{case}_0000..0003.nii.gz` + `labelsTr/{case}.nii.gz`.
- Stage 2: orden de canales = config stage_2.modalities (default t1, t1ce, t2, flair).
- Stage 3: setea nnUNet_raw/preprocessed/results; llama CLI oficial nnU-Net v2.
- Stage 3: default configuration=3d_fullres, fold=0, trainer=nnUNetTrainer.
- Stage 3: device=auto (cuda si hay GPU; avisar si cae a cpu).
- NUNCA modificar Task01_BrainTumour/ ni output/stage_0/ como input mutable.
- NUNCA interpolar seg con linear; SIEMPRE nearest.
- Clases Decathlon nativas {0,1,2,3}. NO remapear en Stage 0/1/2.
- Subregiones: WT [1,2,3], TC [2,3], ET [3].
- Canales Task01: 0=FLAIR, 1=T1, 2=T1ce, 3=T2.
- config.yaml es la única fuente de verdad.

## TDD
Tests unitarios en tests/unit/ para funciones críticas de src/data y src/models.
