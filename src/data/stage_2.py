"""Stage 2: Partición reproducible train/val/test sobre los 484 casos con labels.

Entrada: casos SUCCESS_PREPROCESSED de output/stage_1/.
Salida: output/stage_2/split.json + split_log.csv.

Default: 70% train, 15% val (CV), 15% test (hold-out con labels).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dicom_geometry import ensure_dir
from src.data.stage_0 import project_root_from_config
from src.data.stage_1 import load_stage1_approved_case_ids, stage1_paths

STATUS_OK = "SUCCESS_SPLIT"
STATUS_FAIL = "FAILED"


def stage2_paths(config: dict) -> dict[str, Path]:
    root = project_root_from_config(config)
    cfg = config["stage_2"]
    s1 = stage1_paths(config)

    def _p(key: str, default: str) -> Path:
        value = Path(cfg.get(key, default))
        return value if value.is_absolute() else (root / value)

    return {
        "root": root,
        "input_log": s1["output_log"],
        "output_log": _p("output_log", "output/stage_2/split_log.csv"),
        "split_json": _p("split_json", "output/stage_2/split.json"),
        "audit_report": _p("audit_report", "output/stage_2/reporte_auditoria_stage_2.md"),
    }


def split_ratios_from_config(config: dict) -> dict[str, float]:
    cfg = config.get("stage_2", {})
    train = float(cfg.get("train_ratio", 0.70))
    val = float(cfg.get("val_ratio", 0.15))
    test = float(cfg.get("test_ratio", 0.15))
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios train+val+test deben sumar 1.0, got {total}")
    return {"train": train, "val": val, "test": test}


def split_case_ids(
    case_ids: list[str],
    *,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, list[str]]:
    """Partición reproducible: ordena, mezcla con seed fija, divide por ratios."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio debe ser 1.0")

    unique = sorted({str(c).strip() for c in case_ids if str(c).strip()})
    n = len(unique)
    if n < 3:
        raise ValueError(f"Se necesitan al menos 3 casos para split 70/15/15, got {n}")

    rng = np.random.default_rng(int(seed))
    shuffled = unique.copy()
    rng.shuffle(shuffled)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    assert len(train) + len(val) + len(test) == n
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0

    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }


def build_split_payload(
    case_ids: list[str],
    *,
    seed: int,
    ratios: dict[str, float],
) -> dict:
    parts = split_case_ids(
        case_ids,
        seed=seed,
        train_ratio=ratios["train"],
        val_ratio=ratios["val"],
        test_ratio=ratios["test"],
    )
    n = len(case_ids)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(seed),
        "ratios": ratios,
        "total": n,
        "counts": {k: len(v) for k, v in parts.items()},
        "train": parts["train"],
        "val": parts["val"],
        "test": parts["test"],
    }


def write_split_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def load_split_json(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró split.json: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("train", "val", "test"):
        if key not in data or not isinstance(data[key], list):
            raise ValueError(f"split.json inválido: falta lista '{key}'")
    return data


def verify_split(payload: dict) -> tuple[str, str, dict]:
    """Gate Stage 2: partición sin solapamiento y conteos coherentes."""
    train = list(payload.get("train", []))
    val = list(payload.get("val", []))
    test = list(payload.get("test", []))
    total = int(payload.get("total", len(train) + len(val) + len(test)))

    if not train or not val or not test:
        return STATUS_FAIL, "empty_partition", {"train": len(train), "val": len(val), "test": len(test)}

    all_ids = train + val + test
    if len(set(all_ids)) != len(all_ids):
        return STATUS_FAIL, "duplicate_case_ids", {"n_unique": len(set(all_ids)), "n_total": len(all_ids)}

    if len(all_ids) != total:
        return STATUS_FAIL, "count_mismatch", {"expected": total, "got": len(all_ids)}

    ratios = payload.get("ratios", {})
    return (
        STATUS_OK,
        "",
        {
            "total": total,
            "counts": {"train": len(train), "val": len(val), "test": len(test)},
            "ratios": ratios,
            "seed": payload.get("seed"),
        },
    )


def run_stage_2(config: dict, *, yes: bool = False) -> pd.DataFrame:
    """Orquesta Stage 2: split reproducible 70/15/15 sobre Stage 1."""
    paths = stage2_paths(config)
    cfg = config["stage_2"]
    project = config.get("project", {})
    seed = int(cfg.get("seed", project.get("seed", 42)))
    ratios = split_ratios_from_config(config)
    max_cases = cfg.get("max_cases")

    case_ids = load_stage1_approved_case_ids(paths["input_log"])
    if max_cases is not None:
        case_ids = case_ids[: int(max_cases)]

    print("[*] Stage 2: Partición reproducible train/val/test")
    print(f"    input log:  {paths['input_log']}")
    print(f"    casos:      {len(case_ids)} (aprobados Stage 1)")
    print(f"    ratios:     train={ratios['train']:.0%} val={ratios['val']:.0%} test={ratios['test']:.0%}")
    print(f"    seed:       {seed}")
    print(f"    split.json: {paths['split_json']}")

    if not yes:
        answer = input("Continuar? [y/N]: ").strip().lower()
        if answer not in {"y", "yes", "s", "si"}:
            print("[-] Cancelado por el usuario.")
            return pd.DataFrame()

    payload = build_split_payload(case_ids, seed=seed, ratios=ratios)
    write_split_json(paths["split_json"], payload)
    status, reason, details = verify_split(payload)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "reason": reason,
        "seed": seed,
        "total": payload["total"],
        "n_train": payload["counts"]["train"],
        "n_val": payload["counts"]["val"],
        "n_test": payload["counts"]["test"],
        "train_ratio": ratios["train"],
        "val_ratio": ratios["val"],
        "test_ratio": ratios["test"],
        "split_json": str(paths["split_json"]),
        "details": str(details),
    }
    log_df = pd.DataFrame([record])
    ensure_dir(paths["output_log"].parent)
    log_df.to_csv(paths["output_log"], index=False)

    print(f"[*] Split: train={record['n_train']} val={record['n_val']} test={record['n_test']}")
    print(f"[*] Log: {paths['output_log']}")
    print(f"[*] split.json: {paths['split_json']}")
    print(f"[*] Resumen: {status}" + (f" ({reason})" if reason else ""))
    return log_df
