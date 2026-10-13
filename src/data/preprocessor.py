"""Separación de volúmenes 4D Decathlon → modalidades 3D (Stage 0).

Medical Segmentation Decathlon Task01_BrainTumour:
  modality 0=FLAIR, 1=T1w, 2=t1gd(T1ce), 3=T2w

Detecta automáticamente si el eje de canales está en GetSize()[-1] o [0].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.data.dicom_geometry import read_nifti, write_nifti

# Orden oficial dataset.json Task01
DEFAULT_CHANNEL_TO_MODALITY = {
    0: "flair",
    1: "t1",
    2: "t1ce",
    3: "t2",
}


def load_dataset_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_training_cases(
    decathlon_root: Path,
    dataset_json: Path | None = None,
) -> list[dict]:
    """Lista pares image/label de training.

    Returns:
        lista de dicts: case_id, image_path, label_path
    """
    root = Path(decathlon_root)
    js_path = dataset_json or (root / "dataset.json")
    cases: list[dict] = []

    if js_path.exists():
        meta = load_dataset_json(js_path)
        for entry in meta.get("training", []):
            image_rel = entry["image"].replace("./", "")
            label_rel = entry["label"].replace("./", "")
            image_path = root / image_rel
            label_path = root / label_rel
            case_id = image_path.stem.replace(".nii", "")
            cases.append(
                {
                    "case_id": case_id,
                    "image_path": image_path,
                    "label_path": label_path,
                }
            )
    else:
        images_dir = root / "imagesTr"
        labels_dir = root / "labelsTr"
        for image_path in sorted(images_dir.glob("*.nii.gz")):
            if image_path.name.startswith("._"):
                continue
            case_id = image_path.name.replace(".nii.gz", "")
            label_path = labels_dir / image_path.name
            if label_path.exists():
                cases.append(
                    {
                        "case_id": case_id,
                        "image_path": image_path,
                        "label_path": label_path,
                    }
                )

    cases.sort(key=lambda c: c["case_id"])
    return cases


def list_test_image_paths(decathlon_root: Path, dataset_json: Path | None = None) -> list[dict]:
    """Lista volúmenes 4D de test (Decathlon imagesTs) desde dataset.json."""
    root = Path(decathlon_root)
    js_path = dataset_json or (root / "dataset.json")
    if not js_path.exists():
        return []
    meta = load_dataset_json(js_path)
    entries: list[dict] = []
    for rel in meta.get("test", []):
        image_rel = str(rel).replace("./", "")
        image_path = root / image_rel
        case_id = image_path.stem.replace(".nii", "")
        entries.append({"case_id": case_id, "image_path": image_path})
    entries.sort(key=lambda c: c["case_id"])
    return entries


def detect_channel_axis(size: tuple[int, ...], n_modalities: int = 4) -> int:
    """Detecta el eje de canales en GetSize() de SimpleITK (típicamente el último)."""
    if len(size) != 4:
        raise ValueError(f"Se espera volumen 4D, GetSize={size}")
    matches = [i for i, s in enumerate(size) if s == n_modalities]
    if not matches:
        raise ValueError(
            f"No se encontró eje de {n_modalities} canales en GetSize={size}"
        )
    # Preferir el último eje (convención MSD/SITK: X,Y,Z,C)
    return matches[-1]


def extract_channel_3d(image_4d: sitk.Image, channel: int, channel_axis: int) -> sitk.Image:
    """Extrae un canal 3D del volumen 4D preservando geometría espacial."""
    size = list(image_4d.GetSize())
    index = [0] * image_4d.GetDimension()
    size[channel_axis] = 0
    index[channel_axis] = int(channel)
    extractor = sitk.ExtractImageFilter()
    extractor.SetSize(size)
    extractor.SetIndex(index)
    return extractor.Execute(image_4d)


def split_4d_to_modalities(
    image_4d: sitk.Image,
    channel_to_modality: dict[int, str] | None = None,
) -> dict[str, sitk.Image]:
    """Divide volumen 4D en dict modality -> imagen 3D SITK (intensidades crudas)."""
    mapping = channel_to_modality or DEFAULT_CHANNEL_TO_MODALITY
    # claves pueden venir como str desde YAML
    mapping = {int(k): str(v) for k, v in mapping.items()}
    size = tuple(int(s) for s in image_4d.GetSize())
    axis = detect_channel_axis(size, n_modalities=len(mapping))

    modalities: dict[str, sitk.Image] = {}
    for ch, name in sorted(mapping.items()):
        modalities[name] = extract_channel_3d(image_4d, ch, axis)
    return modalities


def load_and_split_case(
    image_path: Path,
    channel_to_modality: dict[int, str] | None = None,
) -> tuple[dict[str, sitk.Image], sitk.Image]:
    """Carga NIfTI 4D y lo separa en modalidades 3D.

    Returns:
        (modalities_dict, reference_3d) — reference_3d es la primera modalidad
        (geometría espacial nativa a clonar en la máscara si hace falta).
    """
    image_4d = read_nifti(image_path)
    if image_4d.GetDimension() != 4:
        # Algunos lectores pueden devolver vector pixels; intentar vía numpy
        arr = sitk.GetArrayFromImage(image_4d)
        if arr.ndim != 4:
            raise ValueError(
                f"{image_path.name}: se esperaba 4D, dim={image_4d.GetDimension()} arr={arr.shape}"
            )
    modalities = split_4d_to_modalities(image_4d, channel_to_modality)
    ref = next(iter(modalities.values()))
    return modalities, ref


def save_modality_volumes(
    modalities: dict[str, sitk.Image],
    case_id: str,
    volumes_dir: Path,
    modality_order: list[str] | None = None,
) -> dict[str, Path]:
    """Escribe volume_{case_id}-{mod}.nii.gz bajo volumes_dir/volume_{case_id}/."""
    folder_name = f"volume_{case_id}"
    out_dir = volumes_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    order = modality_order or list(modalities.keys())
    paths: dict[str, Path] = {}
    for mod in order:
        path = out_dir / f"{folder_name}-{mod}.nii.gz"
        write_nifti(modalities[mod], path)
        paths[mod] = path
    return paths
