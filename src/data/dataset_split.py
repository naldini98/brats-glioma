"""Partición reproducible train / test para Stage 2 (CV interna: nnU-Net)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.data.nnunet_convert import nnunet_case_id


DEFAULT_RATIOS = {"train": 0.85, "test": 0.15}

SPLIT_NAMES = ("train", "test")


def partition_case_ids(
    case_ids: list[str],
    *,
    ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Partición aleatoria fija en train y test (sin solapamiento)."""
    if not case_ids:
        return {
            "seed": seed,
            "ratios": dict(ratios or DEFAULT_RATIOS),
            "counts": {"train": 0, "test": 0},
            "train": [],
            "test": [],
        }

    r = ratios or DEFAULT_RATIOS
    for key in ("train", "test"):
        if key not in r:
            raise ValueError(f"ratios debe incluir '{key}'")
    total_ratio = float(r["train"]) + float(r["test"])
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"ratios deben sumar 1.0, got {total_ratio}")

    ordered = sorted(case_ids)
    rng = random.Random(seed)
    shuffled = ordered.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * float(r["train"]))
    if n_train > n:
        n_train = n
    n_test = n - n_train

    train = shuffled[:n_train]
    test = shuffled[n_train:]

    assert len(train) + len(test) == n
    assert len(test) == n_test

    return {
        "seed": seed,
        "ratios": {k: float(r[k]) for k in ("train", "test")},
        "counts": {"train": len(train), "test": len(test)},
        "train": sorted(train),
        "test": sorted(test),
    }


def ground_truth_filename(case_id: str) -> str:
    """Nombre del NIfTI de verdad terrenal aislada para test."""
    return f"{nnunet_case_id(case_id)}.nii.gz"


def split_image_label_dirs(split_root: Path, split: str) -> dict[str, Path]:
    """Rutas images*/labels* por partición (convención explícita Stage 2)."""
    root = Path(split_root)
    if split == "train":
        return {"images": root / "imagesTr", "labels": root / "labelsTr"}
    if split == "test":
        return {"images": root / "imagesTs", "labels": root / "labelsTs"}
    raise ValueError(f"split desconocido: {split}")


def expected_case_files_present(
    case_id: str,
    *,
    images_dir: Path,
    labels_dir: Path,
    modalities: list[str],
    label_file: str,
) -> tuple[bool, list[str]]:
    """Comprueba que existen modalidades + seg para un caso en un bundle split."""
    folder = f"volume_{case_id}"
    missing: list[str] = []
    vol_dir = images_dir / folder
    for mod in modalities:
        p = vol_dir / f"{folder}-{mod}.nii.gz"
        if not p.exists():
            missing.append(str(p))
    seg = labels_dir / folder / f"{folder}-{label_file}.nii.gz"
    if not seg.exists():
        missing.append(str(seg))
    return (len(missing) == 0, missing)


def verify_split_bundle(
    manifest: dict[str, Any],
    split: str,
    *,
    split_root: Path,
    modalities: list[str],
    label_file: str,
) -> tuple[bool, str, list[str]]:
    """Verifica que todos los case_id del split tienen imágenes + label emparejados."""
    ids: list[str] = list(manifest.get(split) or [])
    dirs = split_image_label_dirs(split_root, split)
    issues: list[str] = []
    for cid in ids:
        ok, missing = expected_case_files_present(
            cid,
            images_dir=dirs["images"],
            labels_dir=dirs["labels"],
            modalities=modalities,
            label_file=label_file,
        )
        if not ok:
            issues.append(f"{cid}: faltan {len(missing)} archivos")
    if not ids and not split_root.exists():
        return True, "", []
    if issues:
        return False, f"{split}: {len(issues)} casos incompletos", issues
    return True, "", []


def verify_all_split_bundles(
    manifest: dict[str, Any],
    splits_roots: dict[str, Path],
    *,
    modalities: list[str],
    label_file: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        root = splits_roots[split]
        ok, detail, issues = verify_split_bundle(
            manifest,
            split,
            split_root=root,
            modalities=modalities,
            label_file=label_file,
        )
        rows.append(
            {
                "check": f"bundle_{split}",
                "ok": ok,
                "detail": detail or "OK",
                "issues": issues,
                "images": str(split_image_label_dirs(root, split)["images"]),
                "labels": str(split_image_label_dirs(root, split)["labels"]),
            }
        )
    return rows


def write_split_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def load_split_manifest(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró dataset_split.json: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_ids_from_manifest(manifest: dict[str, Any]) -> list[str]:
    """IDs de train; fusiona 'val' legacy en train si existe."""
    train = set(manifest.get("train") or [])
    legacy_val = manifest.get("val")
    if legacy_val:
        train |= set(legacy_val)
    return sorted(train)


def test_ids_from_manifest(manifest: dict[str, Any]) -> list[str]:
    return sorted(manifest.get("test") or [])


def verify_split_disjoint(manifest: dict[str, Any]) -> tuple[bool, str]:
    train = set(train_ids_from_manifest(manifest))
    test = set(manifest.get("test") or [])
    if train & test:
        return False, "solapamiento train/test"
    return True, ""


def verify_split_counts(manifest: dict[str, Any], total_cases: int) -> tuple[bool, str]:
    train = train_ids_from_manifest(manifest)
    test = list(manifest.get("test") or [])
    n = len(set(train)) + len(test)
    if n != total_cases:
        return False, f"suma particiones={n} != total={total_cases}"
    counts = manifest.get("counts") or {}
    if counts.get("train") is not None and int(counts["train"]) != len(manifest.get("train") or []):
        return False, f"counts.train != len(train)"
    if counts.get("test") is not None and int(counts["test"]) != len(test):
        return False, f"counts.test != len(test)"
    return True, ""


def verify_test_ground_truth_dir(
    manifest: dict[str, Any],
    gt_dir: Path,
) -> tuple[bool, str, list[str]]:
    """Comprueba que gt_dir contiene exactamente un archivo por caso test."""
    gt_dir = Path(gt_dir)
    test_ids: list[str] = list(manifest.get("test") or [])
    expected_names = {ground_truth_filename(cid) for cid in test_ids}
    if not gt_dir.exists() and test_ids:
        return False, "test_ground_truth ausente", test_ids

    on_disk = {
        p.name
        for p in gt_dir.glob("*.nii*")
        if p.is_file() and not p.name.startswith("._")
    }
    missing = sorted(expected_names - on_disk)
    extra = sorted(on_disk - expected_names)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"faltan {len(missing)}")
        if extra:
            parts.append(f"sobran {len(extra)}")
        return False, "; ".join(parts), missing + extra
    return True, "", []
