"""Preprocesamiento espacial: reorientación RAS y resampling isotrópico (Stage 1)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import SimpleITK as sitk


def reorient_to_ras(image: sitk.Image) -> sitk.Image:
    """Reorienta volumen 3D a convención RAS (Right-Anterior-Superior)."""
    if image.GetDimension() != 3:
        raise ValueError(f"reorient_to_ras requiere imagen 3D, dim={image.GetDimension()}")
    return sitk.DICOMOrient(image, "RAS")


def resample_to_spacing(
    image: sitk.Image,
    spacing: Sequence[float],
    interpolator: int = sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """Remuestrea imagen 3D a spacing objetivo conservando origen/dirección."""
    if image.GetDimension() != 3:
        raise ValueError(f"resample_to_spacing requiere imagen 3D, dim={image.GetDimension()}")

    original_spacing = np.asarray(image.GetSpacing(), dtype=float)
    original_size = np.asarray(image.GetSize(), dtype=int)
    new_spacing = np.asarray(spacing, dtype=float)
    new_size = np.round(original_size * original_spacing / new_spacing).astype(int)
    new_size = [max(1, int(s)) for s in new_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(s) for s in new_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(default_value))
    return resampler.Execute(image)


def resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    interpolator: int = sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """Remuestrea moving a la grilla exacta de reference (mellizos post-RAS/resample)."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(float(default_value))
    return resampler.Execute(moving)


def build_reference_grid(
    reference_modality: sitk.Image,
    target_spacing: Sequence[float],
) -> sitk.Image:
    """Crea grilla de referencia isotrópica a partir de una modalidad (típicamente t1)."""
    return resample_to_spacing(
        reference_modality,
        spacing=target_spacing,
        interpolator=sitk.sitkLinear,
        default_value=0.0,
    )
