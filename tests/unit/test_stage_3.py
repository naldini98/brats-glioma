"""Tests Stage 3: helpers nnU-Net (sin entrenar)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.models.nnunet_env import (
    apply_nnunet_env,
    detect_device,
    expected_results_dir,
    fold_checkpoint_path,
    parse_folds,
    plans_json_path,
)
from src.models.stage_3 import (
    STATUS_FAIL,
    STATUS_OK,
    STATUS_PLANNED,
    build_plan_command,
    build_train_command,
    stage3_paths,
    verify_fold_checkpoint,
    verify_plans,
    verify_stage2_dataset,
)


def test_parse_folds():
    assert parse_folds([0]) == [0]
    assert parse_folds("all") == [0, 1, 2, 3, 4]
    assert parse_folds(2) == [2]
    assert parse_folds(None) == [0]


def test_apply_nnunet_env(tmp_path: Path):
    raw = tmp_path / "raw"
    pre = tmp_path / "pre"
    res = tmp_path / "res"
    env = apply_nnunet_env(nnunet_raw=raw, nnunet_preprocessed=pre, nnunet_results=res)
    assert Path(env["nnUNet_raw"]) == raw.resolve()
    assert Path(env["nnUNet_preprocessed"]) == pre.resolve()
    assert Path(env["nnUNet_results"]) == res.resolve()


def test_detect_device_explicit_cpu():
    assert detect_device("cpu") == "cpu"


def test_paths_and_checkpoints(tmp_path: Path):
    results = expected_results_dir(
        tmp_path / "results",
        dataset_name="Dataset101_BraTS",
        trainer="nnUNetTrainer",
        plans="nnUNetPlans",
        configuration="3d_fullres",
    )
    ckpt = fold_checkpoint_path(results, 0)
    assert ckpt.name == "checkpoint_final.pth"
    assert "fold_0" in str(ckpt)

    plans = plans_json_path(tmp_path / "pre", "Dataset101_BraTS")
    assert plans.name == "nnUNetPlans.json"


def test_verify_stage2_and_plans(tmp_path: Path):
    ds = tmp_path / "raw" / "Dataset101_BraTS"
    (ds / "imagesTr").mkdir(parents=True)
    (ds / "labelsTr").mkdir(parents=True)
    (ds / "dataset.json").write_text("{}", encoding="utf-8")
    (ds / "labelsTr" / "BRATS_001.nii.gz").write_bytes(b"x" * 200)

    paths = {
        "dataset_json": ds / "dataset.json",
        "images_tr": ds / "imagesTr",
        "labels_tr": ds / "labelsTr",
        "nnunet_preprocessed": tmp_path / "pre",
        "dataset_name": "Dataset101_BraTS",
        "plans": "nnUNetPlans",
        "nnunet_results": tmp_path / "results",
        "trainer": "nnUNetTrainer",
        "configuration": "3d_fullres",
    }
    ok, msg = verify_stage2_dataset(paths)
    assert ok
    assert "1 casos" in msg

    st, reason, _ = verify_plans(paths)
    assert st == STATUS_FAIL
    assert reason == "missing_plans"

    plans = plans_json_path(paths["nnunet_preprocessed"], paths["dataset_name"])
    plans.parent.mkdir(parents=True)
    plans.write_text("{}", encoding="utf-8")
    st, reason, _ = verify_plans(paths)
    assert st == STATUS_PLANNED
    assert reason == ""


def test_verify_fold_checkpoint(tmp_path: Path):
    paths = {
        "nnunet_results": tmp_path / "results",
        "dataset_name": "Dataset101_BraTS",
        "trainer": "nnUNetTrainer",
        "plans": "nnUNetPlans",
        "configuration": "3d_fullres",
    }
    st, reason, _ = verify_fold_checkpoint(paths, 0)
    assert st == STATUS_FAIL

    ckpt = fold_checkpoint_path(
        expected_results_dir(
            paths["nnunet_results"],
            dataset_name=paths["dataset_name"],
            trainer=paths["trainer"],
            plans=paths["plans"],
            configuration=paths["configuration"],
        ),
        0,
    )
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"fake")
    st, reason, det = verify_fold_checkpoint(paths, 0)
    assert st == STATUS_OK
    assert Path(det["checkpoint"]) == ckpt


def test_build_commands(tmp_path: Path):
    paths = {
        "dataset_id": 101,
        "configuration": "3d_fullres",
        "trainer": "nnUNetTrainer",
        "plans": "nnUNetPlans",
    }
    cfg = {
        "verify_dataset_integrity": True,
        "np": 2,
        "configurations": ["3d_fullres"],
    }
    with patch("src.models.stage_3.find_nnunet_command", return_value="nnUNetv2_plan_and_preprocess"):
        cmd = build_plan_command(paths, cfg)
    assert cmd[0] == "nnUNetv2_plan_and_preprocess"
    assert "-d" in cmd and "101" in cmd
    assert "--verify_dataset_integrity" in cmd

    with patch("src.models.stage_3.find_nnunet_command", return_value="nnUNetv2_train"):
        tcmd = build_train_command(paths, {"continue_training": True}, fold=0, device="cpu")
    assert tcmd[0] == "nnUNetv2_train"
    assert "101" in tcmd
    assert "3d_fullres" in tcmd
    assert "0" in tcmd
    assert "--c" in tcmd
    assert "-device" in tcmd and "cpu" in tcmd


def test_stage3_paths_from_config(tmp_path: Path):
    config = {
        "project_root": str(tmp_path),
        "stage_2": {"nnunet_raw": "output/stage_2/nnunet_raw"},
        "stage_3": {
            "nnunet_raw": "output/stage_2/nnunet_raw",
            "nnunet_preprocessed": "output/stage_3/nnunet_preprocessed",
            "nnunet_results": "output/stage_3/nnunet_results",
            "configuration": "3d_fullres",
            "folds": [0],
        },
        "nnunet": {
            "dataset_id": 101,
            "dataset_name": "Dataset101_BraTS",
            "configuration": "3d_fullres",
        },
    }
    paths = stage3_paths(config)
    assert paths["dataset_id"] == 101
    assert paths["dataset_folder"].name == "Dataset101_BraTS"
    assert paths["nnunet_preprocessed"] == tmp_path / "output" / "stage_3" / "nnunet_preprocessed"
