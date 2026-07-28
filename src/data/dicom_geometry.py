"""Geometría SimpleITK y E/S NIfTI para volúmenes mellizos (Stage 0+).

Preserva Spacing / Origin / Direction del volumen fuente (espacio nativo).
En rutas con caracteres no-ASCII (Windows), escribe vía temporal ASCII.
Tras cada escritura verifica que el archivo sea legible (read-after-write).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _needs_ascii_workaround(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def verify_nifti_readable(path: Path, min_bytes: int = 100) -> None:
    """Falla rápido si el NIfTI no se puede leer (permisos / archivo corrupto)."""
    path = Path(path)
    if not path.exists():
        raise OSError(f"No se creó el archivo: {path}")
    size = path.stat().st_size
    if size < min_bytes:
        raise OSError(f"Archivo demasiado pequeño ({size} bytes): {path}")
    with open(path, "rb") as f:
        f.read(1024)
    sitk.ReadImage(str(path))


def write_nifti(image: sitk.Image, path: Path) -> Path:
    """Escribe .nii.gz y verifica lectura inmediata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if _needs_ascii_workaround(path):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / path.name
            sitk.WriteImage(image, str(tmp_path), useCompression=True)
            if path.exists():
                path.unlink()
            shutil.copy2(tmp_path, path)
    else:
        sitk.WriteImage(image, str(path), useCompression=True)

    verify_nifti_readable(path)
    return path


def read_nifti(path: Path) -> sitk.Image:
    """Lee NIfTI; copia a temporal ASCII solo si la ruta tiene caracteres no-ASCII."""
    path = Path(path)

    def _read_from_temp(source: Path) -> sitk.Image:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / source.name
            tmp_path.write_bytes(source.read_bytes())
            return sitk.ReadImage(str(tmp_path))

    if _needs_ascii_workaround(path):
        return _read_from_temp(path)
    try:
        return sitk.ReadImage(str(path))
    except RuntimeError:
        return _read_from_temp(path)


def geometry_tuple(image: sitk.Image) -> dict:
    """Metadatos geométricos 3D comparables."""
    return {
        "size": tuple(int(s) for s in image.GetSize()),
        "spacing": tuple(round(float(s), 6) for s in image.GetSpacing()),
        "origin": tuple(round(float(o), 6) for o in image.GetOrigin()),
        "direction": tuple(round(float(d), 6) for d in image.GetDirection()),
    }


def geometries_match(images: list[sitk.Image]) -> tuple[bool, str]:
    """True si todas comparten size, spacing, origin y direction."""
    if not images:
        return False, "lista de imágenes vacía"
    ref = geometry_tuple(images[0])
    for i, img in enumerate(images[1:], start=1):
        other = geometry_tuple(img)
        for key in ("size", "spacing", "origin", "direction"):
            if other[key] != ref[key]:
                return (
                    False,
                    f"mismatch en {key}: ref={ref[key]} vs img[{i}]={other[key]}",
                )
    return True, "ok"


def apply_geometry(
    image: sitk.Image,
    spacing: Sequence[float],
    origin: Sequence[float],
    direction: Sequence[float],
) -> sitk.Image:
    """Inyecta metadatos geométricos exactos (mellizos)."""
    image.SetSpacing(tuple(float(x) for x in spacing))
    image.SetOrigin(tuple(float(o) for o in origin))
    image.SetDirection(tuple(float(d) for d in direction))
    return image


def copy_geometry_from(reference: sitk.Image, target: sitk.Image) -> sitk.Image:
    """Copia Spacing/Origin/Direction de reference → target."""
    return apply_geometry(
        target,
        spacing=reference.GetSpacing(),
        origin=reference.GetOrigin(),
        direction=reference.GetDirection(),
    )


def array_to_sitk_with_geometry(
    array_zyx: np.ndarray,
    reference: sitk.Image,
    *,
    is_label: bool = False,
) -> sitk.Image:
    """Crea imagen SITK desde array [Z,Y,X] clonando geometría de reference."""
    if array_zyx.ndim != 3:
        raise ValueError(f"Se espera [Z,Y,X], recibido {array_zyx.shape}")
    if is_label:
        arr = np.ascontiguousarray(array_zyx.astype(np.uint8))
        image = sitk.GetImageFromArray(arr)
        image = sitk.Cast(image, sitk.sitkUInt8)
    else:
        arr = np.ascontiguousarray(array_zyx.astype(np.float32))
        image = sitk.GetImageFromArray(arr)
    return copy_geometry_from(reference, image)
