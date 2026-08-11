# NeuroSeg-BraTS Agent Rules

## Pipeline
stage_0 (Decathlon 4D→NIfTI nativo) -> stage_1 (RAS+1mm+z-score) -> stage_2 (split train/test + labels) -> stage_3 (nnU-Net convert) -> stage_4 (plan+train+eval)

## Reglas Críticas
- Stage 0: ESPACIO NATIVO. Sin RAS, resampling ni normalización.
- Stage 1: RAS + resample 1mm + clip percentil + z-score por modalidad/caso.
- Stage 1: NO escalar a [0,1] (MRI no tiene unidades absolutas como HU).
- Stage 2: solo train + test; cada uno con images* + labels* emparejados bajo output/stage_2/splits/.
- Stage 2: train/imagesTr+labelsTr (~85%); test/imagesTs+labelsTs (~15%); test_ground_truth/ copia plana para eval.
- Stage 2: validación cruzada NO aquí; la hace nnU-Net (5-fold) sobre imagesTr en Stage 4.
- Stage 3: train → nnU-Net imagesTr/labelsTr; test → imagesTs sin labels en raw.
- Stage 3: naming `{case}_0000..0003.nii.gz` + `labelsTr/{case}.nii.gz`.
- Stage 3: orden de canales = config stage_3.modalities (default t1, t1ce, t2, flair).
- Stage 4: setea nnUNet_raw/preprocessed/results; llama CLI oficial nnU-Net v2.
- Stage 4: default configuration=3d_fullres, fold=0, trainer=nnUNetTrainer.
- Stage 4: device=auto (cuda si hay GPU; avisar si cae a cpu).
- NUNCA modificar Task01_BrainTumour/ ni output/stage_0/ como input mutable.
- NUNCA interpolar seg con linear; SIEMPRE nearest.
- Clases Decathlon nativas {0,1,2,3}. NO remapear en Stage 0/1/2/3.
- Subregiones: WT [1,2,3], TC [2,3], ET [3].
- Canales Task01: 0=FLAIR, 1=T1, 2=T1ce, 3=T2.
- config.yaml es la única fuente de verdad.

## TDD
Tests unitarios en tests/unit/ para funciones críticas de src/data y src/models.
