#!/usr/bin/env python3
"""Recalcula métricas completas (Dice/IoU/Sens/Spec/HD95) sobre predicciones existentes.

No vuelve a correr inferencia. Usa:
  output/eval/predictions_val/
  output/eval/predictions_test/
  output/stage_1/02_nifti_masks/

Salidas:
  output/eval/full_metrics_log.csv
  output/eval/full_metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline_io import load_config
from src.data.stage_0 import output_paths_for_case
from src.data.stage_1 import stage1_paths
from src.data.stage_2 import load_split_json
from src.data.nnunet_convert import label_filename, nnunet_case_id
from src.data.dicom_geometry import read_nifti
from src.eval.metrics import (
    CLASS_NAMES,
    REGION_NAMES,
    class_metrics,
    flatten_region_metrics,
    region_metrics,
)
from src.eval.run_eval import eval_paths


def spacing_zyx(img: sitk.Image) -> tuple[float, float, float]:
    """SimpleITK spacing (x,y,z) → orden array (z,y,x)."""
    sx, sy, sz = img.GetSpacing()
    return (float(sz), float(sy), float(sx))


def aggregate(records: list[dict], keys: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in keys:
        vals = [float(r[key]) for r in records if r.get("status") == "SUCCESS_EVALUATED" and key in r and np.isfinite(r[key])]
        out[key] = float(np.mean(vals)) if vals else None
        if vals:
            out[f"{key}_median"] = float(np.median(vals))
            out[f"{key}_std"] = float(np.std(vals))
            out[f"{key}_min"] = float(np.min(vals))
            out[f"{key}_max"] = float(np.max(vals))
    return out


def evaluate_subset(
    case_ids: list[str],
    *,
    pred_dir: Path,
    ref_masks_dir: Path,
    label_file: str,
    subset: str,
) -> list[dict]:
    records: list[dict] = []
    n = len(case_ids)
    for i, case_id in enumerate(case_ids, start=1):
        cid = nnunet_case_id(case_id)
        pred_path = pred_dir / label_filename(cid)
        ref_path = output_paths_for_case(
            case_id, Path("_unused"), ref_masks_dir, [], label_file
        )[label_file]
        rec: dict = {
            "case_id": cid,
            "subset": subset,
            "status": "FAILED",
            "reason": "",
            "pred_path": str(pred_path),
            "ref_path": str(ref_path),
        }
        print(f"  [{i}/{n}] {cid} ...", flush=True)
        if not pred_path.exists():
            rec["reason"] = "missing_prediction"
            records.append(rec)
            continue
        if not ref_path.exists():
            rec["reason"] = "missing_reference"
            records.append(rec)
            continue
        try:
            pred_img = read_nifti(pred_path)
            ref_img = read_nifti(ref_path)
            pred_arr = np.asarray(sitk.GetArrayFromImage(pred_img), dtype=np.uint8)
            ref_arr = np.asarray(sitk.GetArrayFromImage(ref_img), dtype=np.uint8)
            spacing = spacing_zyx(ref_img)

            reg = region_metrics(pred_arr, ref_arr, spacing=spacing)
            cls = class_metrics(pred_arr, ref_arr, spacing=spacing)
            flat = flatten_region_metrics(reg)
            rec.update(flat)
            # Compat: columnas cortas Dice
            for region in REGION_NAMES:
                rec[region] = reg[region]["Dice"]
                rec[f"{region}_HD95"] = reg[region]["HD95"]
            for label, name in CLASS_NAMES.items():
                for metric_name, value in cls[name].items():
                    rec[f"class{label}_{metric_name}"] = float(value)
            rec["status"] = "SUCCESS_EVALUATED"
            print(
                f"      Dice WT={reg['WT']['Dice']:.3f} TC={reg['TC']['Dice']:.3f} ET={reg['ET']['Dice']:.3f} | "
                f"HD95 WT={reg['WT']['HD95']:.2f} TC={reg['TC']['HD95']:.2f} ET={reg['ET']['HD95']:.2f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            rec["reason"] = "metric_error"
            rec["details"] = str(exc)
            print(f"      FAIL: {exc}", flush=True)
        records.append(rec)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Métricas completas sobre predicciones existentes")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--subset",
        choices=["val", "test", "both"],
        default="both",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths = eval_paths(config)
    s1 = stage1_paths(config)
    label_file = config.get("stage_3", {}).get("label_file", "seg")

    split = load_split_json(paths["split_json"])
    subsets = ["val", "test"] if args.subset == "both" else [args.subset]

    out_log = paths["output_dir"] / "full_metrics_log.csv"
    out_summary = paths["output_dir"] / "full_metrics_summary.json"
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    print("[*] Métricas completas (Dice / IoU / Sens / Spec / HD95)")
    print(f"    masks: {s1['masks_dir']}")
    print(f"    subsets: {subsets}")

    all_records: list[dict] = []
    subset_summaries: dict = {}
    t0 = time.time()

    metric_keys = (
        [f"{r}_{m}" for r in REGION_NAMES for m in ("Dice", "IoU", "Sensitivity", "Specificity", "HD95")]
        + ["mean_Dice", "mean_HD95"]
        + [f"class{lab}_{m}" for lab in (1, 2, 3) for m in ("Dice", "IoU", "Sensitivity", "Specificity", "HD95")]
    )

    for subset in subsets:
        case_ids = sorted(str(c) for c in split[subset])
        pred_dir = paths["output_dir"] / f"predictions_{subset}"
        print(f"\n[*] {subset}: {len(case_ids)} casos | preds={pred_dir}")
        if not pred_dir.exists():
            raise FileNotFoundError(f"Falta carpeta de predicciones: {pred_dir}. Corré eval primero.")

        records = evaluate_subset(
            case_ids,
            pred_dir=pred_dir,
            ref_masks_dir=s1["masks_dir"],
            label_file=label_file,
            subset=subset,
        )
        all_records.extend(records)
        ok = [r for r in records if r["status"] == "SUCCESS_EVALUATED"]
        agg = aggregate(ok, metric_keys)
        subset_summaries[subset] = {
            "n_cases": len(case_ids),
            "n_ok": len(ok),
            "Dice": {
                "WT": agg.get("WT_Dice"),
                "TC": agg.get("TC_Dice"),
                "ET": agg.get("ET_Dice"),
                "mean": agg.get("mean_Dice"),
            },
            "HD95_mm": {
                "WT": agg.get("WT_HD95"),
                "TC": agg.get("TC_HD95"),
                "ET": agg.get("ET_HD95"),
                "mean": agg.get("mean_HD95"),
            },
            "IoU": {
                "WT": agg.get("WT_IoU"),
                "TC": agg.get("TC_IoU"),
                "ET": agg.get("ET_IoU"),
            },
            "Sensitivity": {
                "WT": agg.get("WT_Sensitivity"),
                "TC": agg.get("TC_Sensitivity"),
                "ET": agg.get("ET_Sensitivity"),
            },
            "Specificity": {
                "WT": agg.get("WT_Specificity"),
                "TC": agg.get("TC_Specificity"),
                "ET": agg.get("ET_Specificity"),
            },
            "classes_Decathlon": {
                "edema_1": {
                    "Dice": agg.get("class1_Dice"),
                    "HD95": agg.get("class1_HD95"),
                },
                "non_enhancing_2": {
                    "Dice": agg.get("class2_Dice"),
                    "HD95": agg.get("class2_HD95"),
                },
                "enhancing_3": {
                    "Dice": agg.get("class3_Dice"),
                    "HD95": agg.get("class3_HD95"),
                },
            },
            "stats": {k: v for k, v in agg.items() if k.endswith(("_median", "_std", "_min", "_max"))},
        }
        d = subset_summaries[subset]["Dice"]
        h = subset_summaries[subset]["HD95_mm"]
        print(
            f"\n    === {subset.upper()} ===\n"
            f"    Dice  WT={d['WT']:.4f}  TC={d['TC']:.4f}  ET={d['ET']:.4f}  mean={d['mean']:.4f}\n"
            f"    HD95  WT={h['WT']:.2f} mm  TC={h['TC']:.2f} mm  ET={h['ET']:.2f} mm  mean={h['mean']:.2f} mm"
        )

    df = pd.DataFrame(all_records)
    df.to_csv(out_log, index=False)

    # También incorporar métricas internas del fold 0 si existen
    fold_summary_path = (
        paths["nnunet_results"]
        / "Dataset101_BraTS"
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        / "fold_0"
        / "validation"
        / "summary.json"
    )
    fold_internal = None
    if fold_summary_path.exists():
        with open(fold_summary_path, encoding="utf-8") as f:
            fold_data = json.load(f)
        fold_internal = {
            "source": str(fold_summary_path),
            "n_cases": 68,
            "note": "Validación interna nnU-Net fold 0 (no es hold-out del proyecto)",
            "foreground_mean_Dice": fold_data.get("foreground_mean", {}).get("Dice"),
            "classes": {
                "edema_1": fold_data.get("mean", {}).get("1", {}).get("Dice"),
                "non_enhancing_2": fold_data.get("mean", {}).get("2", {}).get("Dice"),
                "enhancing_3": fold_data.get("mean", {}).get("3", {}).get("Dice"),
            },
        }

    doc = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - t0, 1),
        "checkpoint": "output/nnunet_results/.../fold_0/checkpoint_final.pth",
        "fold": 0,
        "configuration": "3d_fullres",
        "holdout_subsets": subset_summaries,
        "nnunet_internal_fold0": fold_internal,
        "artifacts": {
            "log_csv": str(out_log),
            "summary_json": str(out_summary),
        },
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n[*] Log: {out_log}")
    print(f"[*] Summary: {out_summary}")
    print(f"[*] Tiempo total: {doc['elapsed_sec']}s")


if __name__ == "__main__":
    main()
