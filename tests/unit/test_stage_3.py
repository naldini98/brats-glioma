"""Tests Stage 3: conversión a formato nnU-Net v2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from src.data.dicom_geometry import write_nifti
from src.data.nnunet_convert import (
    build_dataset_json,
    channel_filename,
    dataset_folder_name,
    label_filename,
    nnunet_case_id,
)
from src.data.stage_3 import STATUS_OK, process_one_case_labeled, verify_stage_3_output


def _make_mri(size=(16, 16, 8), value=1.0, spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    arr = np.full((size[2], size[1], size[0]), value, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img


def _make_seg(size=(16, 16, 8), spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
    arr[2:5, 4:8, 4:8] = 1
    arr[3:6, 6:10, 6:10] = 2
    arr[4:6, 7:11, 7:11] = 3
    img = sitk.GetImageFromArray(arr)
    img = sitk.Cast(img, sitk.sitkUInt8)
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img


@pytest.fixture
def stage1_case_dir(tmp_path: Path):
    case_id = "BRATS_TEST"
    vol_dir = tmp_path / "stage1" / "01_nifti_volumes" / f"volume_{case_id}"
    mask_dir = tmp_path / "stage1" / "02_nifti_masks" / f"volume_{case_id}"
    vol_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for mod, val in [("t1", 0.1), ("t1ce", 0.2), ("t2", 0.3), ("flair", 0.4)]:
        write_nifti(_make_mri(value=val), vol_dir / f"volume_{case_id}-{mod}.nii.gz")
    write_nifti(_make_seg(), mask_dir / f"volume_{case_id}-seg.nii.gz")
    images_tr = tmp_path / "nnunet" / "Dataset101_BraTS" / "imagesTr"
    labels_tr = tmp_path / "nnunet" / "Dataset101_BraTS" / "labelsTr"
    images_tr.mkdir(parents=True)
    labels_tr.mkdir(parents=True)
    return {
        "case_id": case_id,
        "input_volumes": tmp_path / "stage1" / "01_nifti_volumes",
        "input_masks": tmp_path / "stage1" / "02_nifti_masks",
        "images_tr": images_tr,
        "labels_tr": labels_tr,
    }


def test_naming_helpers():
    assert nnunet_case_id("volume_BRATS_001") == "BRATS_001"
    assert nnunet_case_id("BRATS_001") == "BRATS_001"
    assert channel_filename("BRATS_001", 0) == "BRATS_001_0000.nii.gz"
    assert channel_filename("BRATS_001", 3) == "BRATS_001_0003.nii.gz"
    assert label_filename("BRATS_001") == "BRATS_001.nii.gz"
    assert dataset_folder_name(101, "BraTS") == "Dataset101_BraTS"
    assert dataset_folder_name(101, "Dataset101_BraTS") == "Dataset101_BraTS"


def test_build_dataset_json_decathlon_labels():
    meta = build_dataset_json(
        modalities=["t1", "t1ce", "t2", "flair"],
        labels={"background": 0, "edema": 1, "non_enhancing": 2, "enhancing": 3},
        num_training=10,
        dataset_name="Dataset101_BraTS",
    )
    assert meta["channel_names"]["0"] == "t1"
    assert meta["channel_names"]["3"] == "flair"
    assert meta["labels"]["enhancing"] == 3
    assert meta["numTraining"] == 10
    assert "4" not in [str(v) for v in meta["labels"].values()]


def test_process_one_case_stage3(stage1_case_dir):
    d = stage1_case_dir
    modalities = ["t1", "t1ce", "t2", "flair"]
    rec = process_one_case_labeled(
        d["case_id"],
        input_volumes_dir=d["input_volumes"],
        input_masks_dir=d["input_masks"],
        images_tr=d["images_tr"],
        labels_tr=d["labels_tr"],
        modalities=modalities,
        label_file="seg",
    )
    assert rec["status"] == STATUS_OK
    assert (d["images_tr"] / "BRATS_TEST_0000.nii.gz").exists()
    assert (d["images_tr"] / "BRATS_TEST_0003.nii.gz").exists()
    assert (d["labels_tr"] / "BRATS_TEST.nii.gz").exists()

    status, reason, _ = verify_stage_3_output(
        d["case_id"], d["images_tr"], d["labels_tr"], modalities
    )
    assert status == STATUS_OK
    assert reason == ""

    t1 = sitk.ReadImage(str(d["images_tr"] / "BRATS_TEST_0000.nii.gz"))
    arr = np.asarray(sitk.GetArrayFromImage(t1))
    assert float(arr.mean()) == pytest.approx(0.1, abs=1e-5)


def test_dataset_json_roundtrip(tmp_path: Path):
    from src.data.nnunet_convert import write_dataset_json

    meta = build_dataset_json(
        modalities=["t1", "t1ce", "t2", "flair"],
        labels={"background": 0, "edema": 1, "non_enhancing": 2, "enhancing": 3},
        num_training=2,
        dataset_name="Dataset101_BraTS",
    )
    path = write_dataset_json(tmp_path / "dataset.json", meta)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["numTraining"] == 2
    assert loaded["file_ending"] == ".nii.gz"
