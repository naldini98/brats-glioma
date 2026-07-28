import numpy as np

from src.data.labels import invalid_seg_values, to_regions, validate_seg


def test_to_regions_decathlon():
    toy_seg = np.zeros((5, 5, 5), dtype=np.uint8)
    toy_seg[1, 1, 1] = 1  # edema
    toy_seg[2, 2, 2] = 2  # non-enhancing
    toy_seg[3, 3, 3] = 3  # enhancing

    regions = to_regions(toy_seg)
    assert regions["WT"][1, 1, 1] == 1
    assert regions["WT"][2, 2, 2] == 1
    assert regions["WT"][3, 3, 3] == 1
    assert regions["TC"][1, 1, 1] == 0
    assert regions["TC"][2, 2, 2] == 1
    assert regions["TC"][3, 3, 3] == 1
    assert regions["ET"][3, 3, 3] == 1
    assert regions["ET"][2, 2, 2] == 0


def test_validate_seg_accepts_decathlon_classes():
    seg = np.array([0, 1, 2, 3], dtype=np.uint8)
    assert validate_seg(seg) is True
    assert invalid_seg_values(seg) == set()


def test_validate_seg_rejects_class_4():
    seg = np.array([0, 1, 2, 4], dtype=np.uint8)
    assert validate_seg(seg) is False
    assert invalid_seg_values(seg) == {4}


def test_invalid_seg_values_detects_multiple():
    seg = np.array([0, 4, 5, 99], dtype=np.uint8)
    assert invalid_seg_values(seg) == {4, 5, 99}
