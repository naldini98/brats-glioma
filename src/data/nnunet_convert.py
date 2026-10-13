"""Conversión Stage 1 → formato nnU-Net v2 (DatasetXXX_Name).

Naming oficial nnU-Net v2:
  imagesTr/{case_id}_0000.nii.gz ... _{case_id}_000N.nii.gz
  labelsTr/{case_id}.nii.gz
  dataset.json

Clases Decathlon nativas {0,1,2,3}. Sin remap a BraTS clásico {0,1,2,4}.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Sequence

from src.data.dicom_geometry import ensure_dir


def dataset_folder_name(dataset_id: int, dataset_name: str) -> str:
    """Nombre de carpeta DatasetXXX_YYY.

    Acepta dataset_name completo ('Dataset101_BraTS') o sufijo ('BraTS').
    """
    name = str(dataset_name).strip()
    if name.startswith("Dataset") and "_" in name:
        return name
    return f"Dataset{int(dataset_id):03d}_{name}"


def nnunet_case_id(case_id: str) -> str:
    """Identificador limpio para nnU-Net (sin prefijo volume_)."""
    cid = str(case_id).strip()
    if cid.startswith("volume_"):
        cid = cid.replace("volume_", "", 1)
    return cid


def channel_filename(case_id: str, channel_idx: int, file_ending: str = ".nii.gz") -> str:
    """BRATS_001_0000.nii.gz"""
    return f"{nnunet_case_id(case_id)}_{channel_idx:04d}{file_ending}"


def label_filename(case_id: str, file_ending: str = ".nii.gz") -> str:
    """BRATS_001.nii.gz"""
    return f"{nnunet_case_id(case_id)}{file_ending}"


def build_channel_names(modalities: Sequence[str]) -> dict[str, str]:
    """channel_names para dataset.json: {'0': 't1', ...}."""
    return {str(i): str(m) for i, m in enumerate(modalities)}


def build_dataset_json(
    *,
    modalities: Sequence[str],
    labels: dict,
    num_training: int,
    dataset_name: str,
    num_test: int | None = None,
    description: str = "BraTS glioma segmentation (Medical Decathlon Task01)",
    reference: str = "Medical Segmentation Decathlon Task01_BrainTumour",
    licence: str = "CC-BY-SA 4.0",
    release: str = "1.0",
    file_ending: str = ".nii.gz",
) -> dict:
    """Metadatos nnU-Net v2 (labels consecutivos 0..N)."""
    meta = {
        "channel_names": build_channel_names(modalities),
        "labels": {str(k): int(v) for k, v in labels.items()},
        "numTraining": int(num_training),
        "file_ending": file_ending,
        "name": dataset_name,
        "description": description,
        "reference": reference,
        "licence": licence,
        "release": release,
        "tensorImageSize": "3D",
    }
    if num_test is not None:
        meta["numTest"] = int(num_test)
    return meta


def write_dataset_json(path: Path, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def copy_nifti(src: Path, dst: Path) -> Path:
    """Copia NIfTI preservando bytes (sin re-encode)."""
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Origen inexistente: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if not dst.exists() or dst.stat().st_size < 100:
        raise OSError(f"Copia fallida o archivo demasiado pequeño: {dst}")
    return dst


def nnunet_paths_for_case(
    case_id: str,
    images_tr: Path,
    labels_tr: Path,
    n_channels: int,
    file_ending: str = ".nii.gz",
) -> dict[str, Path]:
    """Rutas destino nnU-Net para un caso (training)."""
    paths: dict[str, Path] = {
        f"channel_{i}": images_tr / channel_filename(case_id, i, file_ending)
        for i in range(n_channels)
    }
    paths["label"] = labels_tr / label_filename(case_id, file_ending)
    return paths


def nnunet_test_paths_for_case(
    case_id: str,
    images_ts: Path,
    n_channels: int,
    file_ending: str = ".nii.gz",
) -> dict[str, Path]:
    """Rutas destino nnU-Net para un caso de test (solo canales, sin label)."""
    cid = nnunet_case_id(case_id)
    return {
        f"channel_{i}": images_ts / channel_filename(cid, i, file_ending)
        for i in range(n_channels)
    }


def prepare_dataset_dirs(nnunet_raw: Path, dataset_id: int, dataset_name: str) -> dict[str, Path]:
    """Crea DatasetXXX/imagesTr, labelsTr e imagesTs."""
    folder = ensure_dir(nnunet_raw / dataset_folder_name(dataset_id, dataset_name))
    images_tr = ensure_dir(folder / "imagesTr")
    labels_tr = ensure_dir(folder / "labelsTr")
    images_ts = ensure_dir(folder / "imagesTs")
    return {
        "dataset_folder": folder,
        "images_tr": images_tr,
        "labels_tr": labels_tr,
        "images_ts": images_ts,
        "dataset_json": folder / "dataset.json",
    }
