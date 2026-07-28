"""Tests Stage 1: RAS + resample + clip/z-score."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from src.data.dicom_geometry import write_nifti
from src.data.intensity_norm import clip_and_zscore
from src.data.spatial_preprocess import reorient_to_ras, resample_to_spacing
from src.data.stage_1 import STATUS_OK, process_one_case, verify_stage_1_output


def _make_mri(size=(16, 16, 8), value=100.0, spacing=(2.0, 2.0, 2.0)) -> sitk.Image:
    arr = np.full((size[2], size[1], size[0]), value, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img


def _make_seg(size=(16, 16, 8), spacing=(2.0, 2.0, 2.0)) -> sitk.Image:
    arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
    arr[2:5, 4:8, 4:8] = 1
    arr[3:6, 6:10, 6:10] = 2
    img = sitk.GetImageFromArray(arr)
    img = sitk.Cast(img, sitk.sitkUInt8)
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img


@pytest.fixture
def stage0_case_dir(tmp_path: Path):
    case_id = "BRATS_TEST"
    vol_dir = tmp_path / "stage0" / "01_nifti_volumes" / f"volume_{case_id}"
    mask_dir = tmp_path / "stage0" / "02_nifti_masks" / f"volume_{case_id}"
    vol_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    spacing = (2.0, 2.0, 2.0)
    for mod, val in [("t1", 50.0), ("t1ce", 80.0), ("t2", 120.0), ("flair", 200.0)]:
        write_nifti(_make_mri(value=val, spacing=spacing), vol_dir / f"volume_{case_id}-{mod}.nii.gz")
    write_nifti(_make_seg(spacing=spacing), mask_dir / f"volume_{case_id}-seg.nii.gz")
    return {
        "case_id": case_id,
        "input_volumes": tmp_path / "stage0" / "01_nifti_volumes",
        "input_masks": tmp_path / "stage0" / "02_nifti_masks",
        "output_volumes": tmp_path / "stage1" / "01_nifti_volumes",
        "output_masks": tmp_path / "stage1" / "02_nifti_masks",
    }


def test_reorient_to_ras_keeps_size():
    img = _make_mri()
    ras = reorient_to_ras(img)
    assert ras.GetDimension() == 3
    assert ras.GetSize() == img.GetSize()


def test_resample_to_spacing_halves_voxel_size():
    img = _make_mri(spacing=(2.0, 2.0, 2.0))
    out = resample_to_spacing(img, (1.0, 1.0, 1.0))
    assert tuple(round(s, 4) for s in out.GetSpacing()) == (1.0, 1.0, 1.0)
    assert out.GetSize()[0] >= img.GetSize()[0]


def test_clip_and_zscore_zero_mean_on_brain():
    img = _make_mri(value=50.0)
    arr = np.asarray(sitk.GetArrayFromImage(img))
    arr[2:6, 2:6, 2:6] = 200.0
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.CopyInformation(_make_mri())
    mask = arr > 0
    norm, stats = clip_and_zscore(img, mask)
    norm_arr = np.asarray(sitk.GetArrayFromImage(norm))
    assert norm_arr[mask].mean() == pytest.approx(0.0, abs=1e-4)
    assert norm_arr[mask].std() == pytest.approx(1.0, abs=1e-2)
    assert stats["p_low"] <= stats["p_high"]


def test_process_one_case_stage1(stage0_case_dir):
    d = stage0_case_dir
    rec = process_one_case(
        d["case_id"],
        input_volumes_dir=d["input_volumes"],
        input_masks_dir=d["input_masks"],
        output_volumes_dir=d["output_volumes"],
        output_masks_dir=d["output_masks"],
        modalities=["t1", "t1ce", "t2", "flair"],
        label_file="seg",
        target_spacing=(1.0, 1.0, 1.0),
        reference_modality="t1",
        mask_source="flair",
        mask_threshold=0.0,
        p_low=0.5,
        p_high=99.5,
        eps=1e-8,
    )
    assert rec["status"] == STATUS_OK

    status, reason, _ = verify_stage_1_output(
        d["case_id"],
        d["output_volumes"],
        d["output_masks"],
        target_spacing=(1.0, 1.0, 1.0),
    )
    assert status == STATUS_OK
    assert reason == ""

    t1 = sitk.ReadImage(str(d["output_volumes"] / f"volume_{d['case_id']}" / f"volume_{d['case_id']}-t1.nii.gz"))
    assert tuple(round(s, 4) for s in t1.GetSpacing()) == (1.0, 1.0, 1.0)
