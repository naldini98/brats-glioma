import numpy as np

from src.eval.metrics import (
    dice_binary,
    dice_regions,
    hd95_binary,
    mean_dice,
    region_metrics,
)


def test_dice_binary_perfect_overlap():
    pred = np.array([0, 1, 1, 0], dtype=np.uint8)
    ref = np.array([0, 1, 1, 0], dtype=np.uint8)
    assert dice_binary(pred, ref) == 1.0


def test_dice_binary_both_empty():
    pred = np.zeros(10, dtype=np.uint8)
    ref = np.zeros(10, dtype=np.uint8)
    assert dice_binary(pred, ref) == 1.0


def test_dice_regions_wt_tc_et():
    ref = np.zeros((4, 4, 4), dtype=np.uint8)
    ref[0, 0, 0] = 1
    ref[1, 1, 1] = 2
    ref[2, 2, 2] = 3
    pred = ref.copy()
    scores = dice_regions(pred, ref)
    assert scores == {"WT": 1.0, "TC": 1.0, "ET": 1.0}
    assert mean_dice(scores) == 1.0


def test_dice_regions_partial_et():
    ref = np.zeros((3, 3, 3), dtype=np.uint8)
    ref[1, 1, 1] = 3
    pred = ref.copy()
    pred[1, 1, 2] = 3
    scores = dice_regions(pred, ref)
    assert 0.0 < scores["ET"] < 1.0
    assert scores["WT"] == scores["ET"]


def test_hd95_identical_is_zero():
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[5:10, 5:10, 5:10] = 1
    assert hd95_binary(mask, mask, spacing=(1.0, 1.0, 1.0)) == 0.0


def test_hd95_both_empty_is_zero():
    a = np.zeros((8, 8, 8), dtype=np.uint8)
    assert hd95_binary(a, a) == 0.0


def test_region_metrics_includes_hd95():
    ref = np.zeros((16, 16, 16), dtype=np.uint8)
    ref[4:8, 4:8, 4:8] = 3
    pred = ref.copy()
    scores = region_metrics(pred, ref, spacing=(1.0, 1.0, 1.0))
    assert scores["ET"]["Dice"] == 1.0
    assert scores["ET"]["HD95"] == 0.0
    assert "IoU" in scores["WT"]
