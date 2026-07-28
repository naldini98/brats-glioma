"""Etiquetas Medical Decathlon Task01_BrainTumour (nativas, sin remap).

Clases: 0 fondo, 1 edema, 2 non-enhancing, 3 enhancing.
"""

import numpy as np

VALID_SEG_CLASSES = frozenset({0, 1, 2, 3})


def validate_seg(seg_array: np.ndarray) -> bool:
    """True si todas las etiquetas pertenecen a {0, 1, 2, 3}."""
    return set(np.unique(seg_array)).issubset(VALID_SEG_CLASSES)


def invalid_seg_values(seg_array: np.ndarray) -> set[int]:
    """Valores fuera del esquema Decathlon Task01."""
    return {int(v) for v in np.unique(seg_array)} - VALID_SEG_CLASSES


def to_regions(seg_array: np.ndarray) -> dict:
    """Subregiones jerárquicas sobre clases Decathlon."""
    return {
        "WT": np.where(
            (seg_array == 1) | (seg_array == 2) | (seg_array == 3), 1, 0
        ).astype(np.uint8),
        "TC": np.where((seg_array == 2) | (seg_array == 3), 1, 0).astype(np.uint8),
        "ET": np.where(seg_array == 3, 1, 0).astype(np.uint8),
    }
