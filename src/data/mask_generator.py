"""Máscaras Decathlon Task01: filtro biológico nativo {0, 1, 2, 3}.

Sin remapeo a BraTS clásico {0,1,2,4}.
Clases oficiales dataset.json:
  0 background, 1 edema, 2 non-enhancing tumor, 3 enhancing tumour
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.data.dicom_geometry import copy_geometry_from, read_nifti, write_nifti
from src.data.labels import VALID_SEG_CLASSES, invalid_seg_values, validate_seg


def load_label_as_uint8(label_path: Path) -> sitk.Image:
    """Carga máscara 3D y la fuerza a sitkUInt8."""
    lab = read_nifti(label_path)
    if lab.GetDimension() != 3:
        raise ValueError(f"{label_path.name}: se esperaba máscara 3D, dim={lab.GetDimension()}")
    arr = np.asarray(sitk.GetArrayFromImage(lab))
    arr_u8 = np.ascontiguousarray(arr.astype(np.uint8))
    out = sitk.GetImageFromArray(arr_u8)
    out = sitk.Cast(out, sitk.sitkUInt8)
    return copy_geometry_from(lab, out)


def biological_filter(seg_image: sitk.Image) -> tuple[bool, list[int]]:
    """Valida que solo existan clases {0, 1, 2, 3}."""
    arr = np.asarray(sitk.GetArrayFromImage(seg_image))
    invalid = sorted(invalid_seg_values(arr))
    return validate_seg(arr), invalid


def align_mask_geometry(seg_image: sitk.Image, reference_3d: sitk.Image) -> sitk.Image:
    """Inyecta geometría espacial del volumen de imagen (mellizos)."""
    if tuple(seg_image.GetSize()) != tuple(reference_3d.GetSize()):
        raise ValueError(
            f"Shape máscara {seg_image.GetSize()} != imagen {reference_3d.GetSize()}"
        )
    return copy_geometry_from(reference_3d, seg_image)


def process_and_save_mask(
    label_path: Path,
    case_id: str,
    masks_dir: Path,
    reference_3d: sitk.Image,
    label_file: str = "seg",
) -> tuple[Path, sitk.Image]:
    """Carga, valida biológicamente, alinea geometría y guarda volume_{id}-seg.nii.gz."""
    seg = load_label_as_uint8(label_path)
    ok, invalid = biological_filter(seg)
    if not ok:
        raise ValueError(
            f"Máscara corrupta: valores fuera de {sorted(VALID_SEG_CLASSES)} -> {invalid}"
        )
    seg = align_mask_geometry(seg, reference_3d)

    folder_name = f"volume_{case_id}"
    out_dir = masks_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{folder_name}-{label_file}.nii.gz"
    write_nifti(seg, out_path)
    return out_path, seg
