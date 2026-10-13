"""Métricas BraTS: Dice, IoU, sensibilidad, HD95 sobre Decathlon {0,1,2,3}."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from src.data.labels import to_regions, validate_seg

REGION_NAMES = ("WT", "TC", "ET")
CLASS_NAMES = {1: "edema", 2: "non_enhancing", 3: "enhancing"}


def dice_binary(pred: np.ndarray, ref: np.ndarray) -> float:
    """Dice binario; si ambas máscaras vacías → 1.0 (convención BraTS)."""
    pred_b = pred.astype(bool)
    ref_b = ref.astype(bool)
    intersection = int(np.logical_and(pred_b, ref_b).sum())
    denom = int(pred_b.sum()) + int(ref_b.sum())
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom


def iou_binary(pred: np.ndarray, ref: np.ndarray) -> float:
    """IoU / Jaccard; ambas vacías → 1.0."""
    pred_b = pred.astype(bool)
    ref_b = ref.astype(bool)
    intersection = int(np.logical_and(pred_b, ref_b).sum())
    union = int(np.logical_or(pred_b, ref_b).sum())
    if union == 0:
        return 1.0
    return intersection / union


def sensitivity_binary(pred: np.ndarray, ref: np.ndarray) -> float:
    """Sensibilidad (recall); ref vacía → 1.0 si pred también vacía, else 0.0."""
    pred_b = pred.astype(bool)
    ref_b = ref.astype(bool)
    tp = int(np.logical_and(pred_b, ref_b).sum())
    fn = int(np.logical_and(~pred_b, ref_b).sum())
    denom = tp + fn
    if denom == 0:
        return 1.0 if not pred_b.any() else 0.0
    return tp / denom


def specificity_binary(pred: np.ndarray, ref: np.ndarray) -> float:
    """Especificidad; ambos vacíos de fondo no aplica → 1.0 si no hay FP."""
    pred_b = pred.astype(bool)
    ref_b = ref.astype(bool)
    tn = int(np.logical_and(~pred_b, ~ref_b).sum())
    fp = int(np.logical_and(pred_b, ~ref_b).sum())
    denom = tn + fp
    if denom == 0:
        return 1.0
    return tn / denom


def _border(mask: np.ndarray) -> np.ndarray:
    """Borde binario 3D (superficie)."""
    mask_b = mask.astype(bool)
    if not mask_b.any():
        return mask_b
    eroded = ndimage.binary_erosion(mask_b, iterations=1, border_value=0)
    return mask_b & ~eroded


def hd95_binary(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Hausdorff Distance 95% (mm). Ambas vacías → 0.0; una vacía → diagonal del volumen.

    ``spacing`` en orden del array (z, y, x) — SimpleITK GetArrayFromImage.
    """
    pred_b = pred.astype(bool)
    ref_b = ref.astype(bool)
    if not pred_b.any() and not ref_b.any():
        return 0.0

    sp = tuple(float(s) for s in spacing)
    diag = float(np.linalg.norm(np.asarray(pred.shape, dtype=np.float64) * np.asarray(sp)))

    if not pred_b.any() or not ref_b.any():
        return diag

    pred_border = _border(pred_b)
    ref_border = _border(ref_b)
    if not pred_border.any() or not ref_border.any():
        return diag

    # EDT: distancia (mm) de cada voxel al borde opuesto
    dt_to_ref = ndimage.distance_transform_edt(~ref_border, sampling=sp)
    dt_to_pred = ndimage.distance_transform_edt(~pred_border, sampling=sp)
    distances = np.concatenate([dt_to_ref[pred_border], dt_to_pred[ref_border]])
    return float(np.percentile(distances, 95))


def binary_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, float]:
    return {
        "Dice": dice_binary(pred, ref),
        "IoU": iou_binary(pred, ref),
        "Sensitivity": sensitivity_binary(pred, ref),
        "Specificity": specificity_binary(pred, ref),
        "HD95": hd95_binary(pred, ref, spacing=spacing),
    }


def dice_regions(pred_seg: np.ndarray, ref_seg: np.ndarray) -> dict[str, float]:
    """Dice por región WT/TC/ET."""
    if pred_seg.shape != ref_seg.shape:
        raise ValueError(f"Shape mismatch: pred {pred_seg.shape} vs ref {ref_seg.shape}")
    if not validate_seg(pred_seg):
        raise ValueError("Predicción con etiquetas inválidas")
    if not validate_seg(ref_seg):
        raise ValueError("Referencia con etiquetas inválidas")

    pred_regions = to_regions(pred_seg)
    ref_regions = to_regions(ref_seg)
    return {
        region: dice_binary(pred_regions[region], ref_regions[region])
        for region in REGION_NAMES
    }


def region_metrics(
    pred_seg: np.ndarray,
    ref_seg: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, dict[str, float]]:
    """Métricas completas por región WT/TC/ET."""
    if pred_seg.shape != ref_seg.shape:
        raise ValueError(f"Shape mismatch: pred {pred_seg.shape} vs ref {ref_seg.shape}")
    if not validate_seg(pred_seg):
        raise ValueError("Predicción con etiquetas inválidas")
    if not validate_seg(ref_seg):
        raise ValueError("Referencia con etiquetas inválidas")

    pred_regions = to_regions(pred_seg)
    ref_regions = to_regions(ref_seg)
    return {
        region: binary_metrics(pred_regions[region], ref_regions[region], spacing=spacing)
        for region in REGION_NAMES
    }


def class_metrics(
    pred_seg: np.ndarray,
    ref_seg: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, dict[str, float]]:
    """Métricas por clase Decathlon 1/2/3."""
    if pred_seg.shape != ref_seg.shape:
        raise ValueError(f"Shape mismatch: pred {pred_seg.shape} vs ref {ref_seg.shape}")
    out: dict[str, dict[str, float]] = {}
    for label, name in CLASS_NAMES.items():
        out[name] = binary_metrics(pred_seg == label, ref_seg == label, spacing=spacing)
    return out


def mean_dice(regions: dict[str, float]) -> float:
    """Promedio WT/TC/ET (solo Dice)."""
    return float(np.mean([regions["WT"], regions["TC"], regions["ET"]]))


def flatten_region_metrics(region_scores: dict[str, dict[str, float]]) -> dict[str, float]:
    """Aplana {'WT': {'Dice':..,'HD95':..}, ...} → {'WT_Dice':.., 'WT_HD95':..}."""
    flat: dict[str, float] = {}
    for region, metrics in region_scores.items():
        for metric_name, value in metrics.items():
            flat[f"{region}_{metric_name}"] = float(value)
    # Promedios Dice / HD95
    flat["mean_Dice"] = float(np.mean([region_scores[r]["Dice"] for r in REGION_NAMES]))
    flat["mean_HD95"] = float(np.mean([region_scores[r]["HD95"] for r in REGION_NAMES]))
    return flat
