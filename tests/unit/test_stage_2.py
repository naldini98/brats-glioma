"""Tests Stage 2: split reproducible 70/15/15."""

from __future__ import annotations

import json

import pytest

from src.data.stage_2 import (
    STATUS_OK,
    build_split_payload,
    split_case_ids,
    verify_split,
    write_split_json,
)


def test_split_case_ids_reproducible():
    ids = [f"BRATS_{i:03d}" for i in range(100)]
    a = split_case_ids(ids, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    b = split_case_ids(ids, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert a == b
    assert len(a["train"]) + len(a["val"]) + len(a["test"]) == 100
    assert not set(a["train"]) & set(a["val"])
    assert not set(a["train"]) & set(a["test"])
    assert not set(a["val"]) & set(a["test"])


def test_split_case_ids_different_seed():
    ids = [f"C{i}" for i in range(20)]
    a = split_case_ids(ids, seed=1)
    b = split_case_ids(ids, seed=2)
    assert a != b


def test_split_484_counts():
    ids = [f"BRATS_{i:03d}" for i in range(484)]
    parts = split_case_ids(ids, seed=42)
    assert len(parts["train"]) == 338
    assert len(parts["val"]) == 72
    assert len(parts["test"]) == 74


def test_verify_split_ok():
    payload = build_split_payload(
        [f"C{i}" for i in range(10)],
        seed=42,
        ratios={"train": 0.7, "val": 0.15, "test": 0.15},
    )
    status, reason, _ = verify_split(payload)
    assert status == STATUS_OK
    assert reason == ""


def test_write_split_json_roundtrip(tmp_path):
    payload = build_split_payload(["A", "B", "C", "D", "E", "F"], seed=7, ratios={
        "train": 0.5, "val": 0.25, "test": 0.25
    })
    path = write_split_json(tmp_path / "split.json", payload)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["seed"] == 7
    assert "train" in loaded and "val" in loaded and "test" in loaded


def test_split_ratios_must_sum_one():
    with pytest.raises(ValueError):
        split_case_ids(["a", "b", "c"], seed=1, train_ratio=0.5, val_ratio=0.3, test_ratio=0.1)


def test_load_stage1_approved_case_ids(tmp_path):
    from src.data.stage_1 import load_stage1_approved_case_ids

    log = tmp_path / "preprocessing_log.csv"
    log.write_text(
        "case_id,status\n"
        "BRATS_001,SUCCESS_PREPROCESSED\n"
        "BRATS_002,FAILED\n"
        "BRATS_003,SUCCESS_PREPROCESSED\n",
        encoding="utf-8",
    )
    ids = load_stage1_approved_case_ids(log)
    assert ids == ["BRATS_001", "BRATS_003"]
