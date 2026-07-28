"""Tests Stage 0 Decathlon: 4D → modalidades 3D + seg {0,1,2,3}."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from src.data.dicom_geometry import geometries_match, write_nifti
from src.data.preprocessor import detect_channel_axis, split_4d_to_modalities
from src.data.stage_0 import STATUS_OK, process_one_case, verify_stage_0_output


def _make_4d_image(spatial=(8, 8, 6), n_channels=4) -> sitk.Image:
    """Crea volumen 4D SITK size (X,Y,Z,C) con geometría no trivial."""
    x, y, z = spatial
    # Array numpy para SITK 4D: (C, Z, Y, X)
    arr = np.zeros((n_channels, z, y, x), dtype=np.float32)
    for c in range(n_channels):
        arr[c] = float(c + 1)
    img = sitk.GetImageFromArray(arr, isVector=False)
    # GetImageFromArray on 4D may produce size (X,Y,Z,C)
    img.SetSpacing((1.0, 1.0, 1.25, 1.0))
    img.SetOrigin((10.0, 20.0, 30.0, 0.0))
    # Direction 4x4 identidad
    img.SetDirection(tuple(np.eye(4).ravel().tolist()))
    return img


def _make_label(spatial=(8, 8, 6)) -> sitk.Image:
    x, y, z = spatial
    arr = np.zeros((z, y, x), dtype=np.uint8)
    arr[1:3, 2:4, 2:4] = 1
    arr[2:4, 4:6, 4:6] = 2
    arr[3:5, 1:3, 1:3] = 3
    lab = sitk.GetImageFromArray(arr)
    lab = sitk.Cast(lab, sitk.sitkUInt8)
    lab.SetSpacing((1.0, 1.0, 1.25))
    lab.SetOrigin((10.0, 20.0, 30.0))
    lab.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return lab


@pytest.fixture
def synthetic_case(tmp_path: Path):
    images_dir = tmp_path / "imagesTr"
    labels_dir = tmp_path / "labelsTr"
    images_dir.mkdir()
    labels_dir.mkdir()
    case_id = "BRATS_999"
    img_path = images_dir / f"{case_id}.nii.gz"
    lab_path = labels_dir / f"{case_id}.nii.gz"
    write_nifti(_make_4d_image(), img_path)
    write_nifti(_make_label(), lab_path)
    return {
        "case_id": case_id,
        "image_path": img_path,
        "label_path": lab_path,
        "root": tmp_path,
    }


def test_detect_channel_axis_last():
    assert detect_channel_axis((240, 240, 155, 4)) == 3


def test_detect_channel_axis_first():
    assert detect_channel_axis((4, 240, 240, 155)) == 0


def test_split_4d_modalities_names():
    img = _make_4d_image()
    mapping = {0: "flair", 1: "t1", 2: "t1ce", 3: "t2"}
    mods = split_4d_to_modalities(img, mapping)
    assert set(mods) == {"flair", "t1", "t1ce", "t2"}
    for m in mods.values():
        assert m.GetDimension() == 3
        assert m.GetSize() == (8, 8, 6)


def test_process_one_case_success(synthetic_case, tmp_path):
    volumes_dir = tmp_path / "01_nifti_volumes"
    masks_dir = tmp_path / "02_nifti_masks"
    modalities = ["t1", "t1ce", "t2", "flair"]
    channel_map = {0: "flair", 1: "t1", 2: "t1ce", 3: "t2"}

    rec = process_one_case(
        synthetic_case,
        volumes_dir=volumes_dir,
        masks_dir=masks_dir,
        modalities=modalities,
        label_file="seg",
        channel_to_modality=channel_map,
    )
    assert rec["status"] == STATUS_OK
    assert rec["shapes"] == "6x8x8"

    status, reason, _ = verify_stage_0_output(
        synthetic_case["case_id"], volumes_dir, masks_dir, modalities, "seg"
    )
    assert status == STATUS_OK
    assert reason == ""

    t1 = sitk.ReadImage(
        str(volumes_dir / "volume_BRATS_999" / "volume_BRATS_999-t1.nii.gz")
    )
    seg = sitk.ReadImage(
        str(masks_dir / "volume_BRATS_999" / "volume_BRATS_999-seg.nii.gz")
    )
    assert seg.GetPixelID() == sitk.sitkUInt8
    ok, _ = geometries_match([t1, seg])
    assert ok


def test_verify_fails_if_missing(tmp_path):
    status, reason, _ = verify_stage_0_output("BRATS_X", tmp_path / "v", tmp_path / "m")
    assert status == "FAILED"
    assert reason == "missing_nifti"
