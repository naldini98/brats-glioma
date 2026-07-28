"""Normalización de intensidades MRI: clip percentil + z-score por modalidad/caso (Stage 1).

MRI no tiene unidades absolutas (a diferencia de HU en TAC). Se normaliza por
modalidad y por paciente sobre voxeles cerebrales, sin escalar a [0,1].
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from src.data.dicom_geometry import copy_geometry_from


def brain_mask_from_modality(
    image: sitk.Image,
    *,
    threshold: float = 0.0,
) -> np.ndarray:
    """Máscara booleana de voxeles cerebrales (intensidad > umbral)."""
    arr = np.asarray(sitk.GetArrayFromImage(image), dtype=np.float32)
    return arr > threshold


def clip_and_zscore(
    image: sitk.Image,
    mask: np.ndarray,
    *,
    p_low: float = 0.5,
    p_high: float = 99.5,
    eps: float = 1e-8,
) -> tuple[sitk.Image, dict]:
    """Clip percentil + z-score sobre voxeles en mask.

    Returns:
        (normalized_image, stats_dict) con p_low, p_high, mean, std.
    """
    arr = np.asarray(sitk.GetArrayFromImage(image), dtype=np.float32)
    if arr.shape != mask.shape:
        raise ValueError(f"mask shape {mask.shape} != image shape {arr.shape}")

    brain = arr[mask]
    stats = {"p_low": float("nan"), "p_high": float("nan"), "mean": float("nan"), "std": float("nan")}

    if brain.size == 0:
        out = sitk.GetImageFromArray(arr.astype(np.float32))
        return copy_geometry_from(image, out), stats

    lo = float(np.percentile(brain, p_low))
    hi = float(np.percentile(brain, p_high))
    clipped = np.clip(arr, lo, hi)
    brain_clipped = clipped[mask]
    mean = float(brain_clipped.mean())
    std = float(brain_clipped.std())
    if std < eps:
        std = eps

    normalized = ((clipped - mean) / std).astype(np.float32)
    out = sitk.GetImageFromArray(normalized)
    out = copy_geometry_from(image, out)

    stats.update({"p_low": lo, "p_high": hi, "mean": mean, "std": std})
    return out, stats


def normalize_modality_volumes(
    modalities: dict[str, sitk.Image],
    *,
    mask_source: str = "flair",
    mask_threshold: float = 0.0,
    p_low: float = 0.5,
    p_high: float = 99.5,
    eps: float = 1e-8,
) -> tuple[dict[str, sitk.Image], dict[str, dict]]:
    """Normaliza cada modalidad con la misma máscara cerebral (de mask_source)."""
    if mask_source not in modalities:
        raise KeyError(f"Modalidad para máscara no encontrada: {mask_source}")

    mask = brain_mask_from_modality(modalities[mask_source], threshold=mask_threshold)
    normalized: dict[str, sitk.Image] = {}
    stats_by_mod: dict[str, dict] = {}

    for name, img in modalities.items():
        norm_img, stats = clip_and_zscore(
            img, mask, p_low=p_low, p_high=p_high, eps=eps
        )
        normalized[name] = norm_img
        stats_by_mod[name] = stats

    return normalized, stats_by_mod
