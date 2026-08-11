"""Tests Stage 2: split train/test + imágenes con labels."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.dataset_split import (
    ground_truth_filename,
    partition_case_ids,
    split_image_label_dirs,
    verify_split_counts,
    verify_split_disjoint,
    verify_test_ground_truth_dir,
)
from src.data.dicom_geometry import write_nifti
from src.data.stage_2 import STATUS_OK, copy_test_ground_truth_flat, run_stage_2


def _tiny_mri():
    import SimpleITK as sitk
    import numpy as np

    size = (16, 16, 8)
    arr = np.zeros((size[2], size[1], size[0]), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def _tiny_seg():
    import SimpleITK as sitk
    import numpy as np

    size = (16, 16, 8)
    arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
    arr[2:5, 4:8, 4:8] = 1
    img = sitk.GetImageFromArray(arr)
    img = sitk.Cast(img, sitk.sitkUInt8)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def test_partition_disjoint_and_sum():
    ids = [f"BRATS_{i:03d}" for i in range(100)]
    manifest = partition_case_ids(ids, seed=42)
    ok_d, _ = verify_split_disjoint(manifest)
    ok_c, _ = verify_split_counts(manifest, 100)
    assert ok_d
    assert ok_c
    assert manifest["counts"]["train"] + manifest["counts"]["test"] == 100


def test_partition_reproducible():
    ids = [f"BRATS_{i:03d}" for i in range(50)]
    a = partition_case_ids(ids, seed=42)
    b = partition_case_ids(ids, seed=42)
    assert a["train"] == b["train"]
    assert a["test"] == b["test"]


def test_484_train_test_ratios():
    ids = [f"BRATS_{i:03d}" for i in range(484)]
    m = partition_case_ids(ids, seed=42, ratios={"train": 0.85, "test": 0.15})
    assert m["counts"]["train"] in {411, 412}
    assert m["counts"]["test"] in {72, 73}


def test_stage2_paths_minimal_config(tmp_path: Path):
    """config.yaml parcial (sin claves nuevas de split) debe resolver rutas."""
    from src.data.stage_2 import stage2_paths

    config = {
        "project_root": str(tmp_path),
        "stage_0": {
            "decathlon_root": "Task01_BrainTumour",
            "output_log": "output/stage_0/processing_log.csv",
            "volumes_dir": "output/stage_0/01_nifti_volumes",
            "masks_dir": "output/stage_0/02_nifti_masks",
            "audit_report": "output/stage_0/reporte_auditoria_stage_0.md",
        },
        "stage_1": {
            "output_log": "output/stage_1/preprocessing_log.csv",
            "volumes_dir": "output/stage_1/01_nifti_volumes",
            "masks_dir": "output/stage_1/02_nifti_masks",
            "audit_report": "output/stage_1/reporte_auditoria_stage_1.md",
        },
        "stage_2": {},
    }
    paths = stage2_paths(config)
    assert paths["split_manifest"] == tmp_path / "output/stage_2/dataset_split.json"
    assert paths["splits_dir"] == tmp_path / "output/stage_2/splits"


def test_stage2_rejects_obsolete_nnunet_raw():
    from src.data.stage_2 import stage2_paths

    config = {
        "project_root": "C:/tmp",
        "stage_0": {
            "decathlon_root": "Task01_BrainTumour",
            "output_log": "output/stage_0/processing_log.csv",
            "volumes_dir": "output/stage_0/01_nifti_volumes",
            "masks_dir": "output/stage_0/02_nifti_masks",
            "audit_report": "output/stage_0/reporte_auditoria_stage_0.md",
        },
        "stage_1": {
            "output_log": "output/stage_1/preprocessing_log.csv",
            "volumes_dir": "output/stage_1/01_nifti_volumes",
            "masks_dir": "output/stage_1/02_nifti_masks",
            "audit_report": "output/stage_1/reporte_auditoria_stage_1.md",
        },
        "stage_2": {"nnunet_raw": "output/stage_2/nnunet_raw"},
    }
    import pytest

    with pytest.raises(ValueError, match="obsoletas"):
        stage2_paths(config)


def test_copy_test_ground_truth(tmp_path: Path):
    case_id = "BRATS_099"
    labels_ts = tmp_path / "test" / "labelsTs"
    folder = f"volume_{case_id}"
    mask_dir = labels_ts / folder
    mask_dir.mkdir(parents=True)

    write_nifti(_tiny_seg(), mask_dir / f"{folder}-seg.nii.gz")
    gt_dir = tmp_path / "gt"
    status, reason, path = copy_test_ground_truth_flat(
        case_id, labels_ts_dir=labels_ts, gt_dir=gt_dir, label_file="seg"
    )
    assert status == STATUS_OK
    assert reason == ""
    assert Path(path).name == ground_truth_filename(case_id)


def test_run_stage_2_integration(tmp_path: Path):
    """Pipeline split con log Stage 1 sintético."""
    import pandas as pd

    project = tmp_path
    for i in range(10):
        cid = f"BRATS_{i:03d}"
        vdir = project / "output/stage_1/01_nifti_volumes/volume_{}".format(cid)
        mdir = project / "output/stage_1/02_nifti_masks/volume_{}".format(cid)
        vdir.mkdir(parents=True)
        mdir.mkdir(parents=True)
        for mod in ("t1", "t1ce", "t2", "flair"):
            write_nifti(_tiny_mri(), vdir / f"volume_{cid}-{mod}.nii.gz")
        write_nifti(_tiny_seg(), mdir / f"volume_{cid}-seg.nii.gz")

    log = project / "output/stage_1/preprocessing_log.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"case_id": [f"BRATS_{i:03d}" for i in range(10)], "status": ["SUCCESS_PREPROCESSED"] * 10}
    ).to_csv(log, index=False)

    config = {
        "project_root": str(project),
        "project": {"seed": 42},
        "stage_0": {
            "decathlon_root": "Task01_BrainTumour",
            "output_log": "output/stage_0/processing_log.csv",
            "volumes_dir": "output/stage_0/01_nifti_volumes",
            "masks_dir": "output/stage_0/02_nifti_masks",
            "audit_report": "output/stage_0/reporte_auditoria_stage_0.md",
        },
        "stage_1": {
            "modalities": ["t1", "t1ce", "t2", "flair"],
            "output_log": "output/stage_1/preprocessing_log.csv",
            "volumes_dir": "output/stage_1/01_nifti_volumes",
            "masks_dir": "output/stage_1/02_nifti_masks",
            "audit_report": "output/stage_1/reporte_auditoria_stage_1.md",
        },
        "stage_2": {
            "output_log": "output/stage_2/processing_log.csv",
            "audit_report": "output/stage_2/reporte_auditoria_stage_2.md",
            "split_manifest": "output/stage_2/dataset_split.json",
            "splits_dir": "output/stage_2/splits",
            "test_ground_truth_dir": "output/stage_2/test_ground_truth",
            "seed": 42,
            "ratios": {"train": 0.85, "test": 0.15},
            "label_file": "seg",
        },
    }
    run_stage_2(config, yes=True)
    manifest = json.loads(
        (project / "output/stage_2/dataset_split.json").read_text(encoding="utf-8")
    )
    train_dirs = split_image_label_dirs(project / "output/stage_2/splits/train", "train")
    sample_train = manifest["train"][0]
    assert (train_dirs["images"] / f"volume_{sample_train}").exists()
    assert (
        train_dirs["labels"]
        / f"volume_{sample_train}"
        / f"volume_{sample_train}-seg.nii.gz"
    ).exists()
    test_dirs = split_image_label_dirs(project / "output/stage_2/splits/test", "test")
    sample_test = manifest["test"][0]
    assert (test_dirs["images"] / f"volume_{sample_test}").exists()
    assert (
        test_dirs["labels"]
        / f"volume_{sample_test}"
        / f"volume_{sample_test}-seg.nii.gz"
    ).exists()
    ok_gt, _, _ = verify_test_ground_truth_dir(
        manifest, project / "output/stage_2/test_ground_truth"
    )
    assert ok_gt
