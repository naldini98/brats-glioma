"""Fixtures compartidas para pytest."""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def config():
    config_path = ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def brats_classes():
    """Clases Decathlon Task01 nativas."""
    return {0, 1, 2, 3}


@pytest.fixture
def default_affine():
    return np.diag([1.0, 1.0, 1.0, 1.0])


def write_nifti(path: Path, shape=(10, 10, 10), affine=None, data=None, dtype=np.float32) -> Path:
    """Escribe un .nii.gz sintético para tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.asarray(affine if affine is not None else np.eye(4))
    data = data if data is not None else np.zeros(shape, dtype=dtype)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


@pytest.fixture
def write_nifti_fixture():
    return write_nifti
