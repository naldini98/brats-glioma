"""Evaluación hold-out: inferencia nnU-Net + Dice WT/TC/ET en val/test."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.data.dicom_geometry import ensure_dir, read_nifti
from src.data.nnunet_convert import copy_nifti, label_filename, nnunet_case_id, nnunet_test_paths_for_case
from src.data.stage_0 import output_paths_for_case, project_root_from_config
from src.data.stage_1 import stage1_paths
from src.data.stage_2 import load_split_json, stage2_paths as split_stage2_paths
from src.eval.metrics import dice_regions, mean_dice
from src.models.nnunet_env import (
    apply_nnunet_env,
    detect_device,
    expected_results_dir,
    find_nnunet_command,
    fold_checkpoint_path,
    parse_folds,
    resolve_under_root,
    run_command,
)
from src.models.stage_4 import STATUS_OK as TRAIN_OK, stage4_paths, verify_fold_checkpoint

STATUS_OK = "SUCCESS_EVALUATED"
STATUS_FAIL = "FAILED"


def eval_paths(config: dict) -> dict:
    root = project_root_from_config(config)
    cfg = config.get("eval", {})
    s1 = stage1_paths(config)
    s2 = split_stage2_paths(config)
    s4 = stage4_paths(config)

    def _p(key: str, default: str) -> Path:
        value = cfg.get(key, default)
        return resolve_under_root(root, value)

    output_dir = _p("output_dir", "output/eval")

    return {
        "root": root,
        "split_json": s2["split_json"],
        "input_volumes_dir": s1["volumes_dir"],
        "input_masks_dir": s1["masks_dir"],
        "output_dir": output_dir,
        "output_log": _p("output_log", "output/eval/eval_log.csv"),
        "summary_json": _p("summary_json", "output/eval/summary.json"),
        "nnunet_raw": s4["nnunet_raw"],
        "nnunet_preprocessed": s4["nnunet_preprocessed"],
        "nnunet_results": s4["nnunet_results"],
        "dataset_id": s4["dataset_id"],
        "dataset_name": s4["dataset_name"],
        "configuration": s4["configuration"],
        "trainer": s4["trainer"],
        "plans": s4["plans"],
    }


def load_eval_subsets(split_json: Path, subsets: list[str]) -> dict[str, list[str]]:
    data = load_split_json(split_json)
    out: dict[str, list[str]] = {}
    for name in subsets:
        if name not in data:
            raise KeyError(f"Subset '{name}' no existe en {split_json}")
        out[name] = sorted(str(c) for c in data[name])
    return out


def prepare_eval_input(
    case_ids: list[str],
    *,
    input_volumes_dir: Path,
    eval_input_dir: Path,
    modalities: list[str],
) -> tuple[list[str], list[str]]:
    """Copia volúmenes Stage 1 al formato nnU-Net (_0000.._0003)."""
    ensure_dir(eval_input_dir)
    ok_ids: list[str] = []
    failed: list[str] = []

    for case_id in case_ids:
        cid = nnunet_case_id(case_id)
        in_paths = output_paths_for_case(case_id, input_volumes_dir, Path("_unused"), modalities, "seg")
        missing = [mod for mod in modalities if not in_paths[mod].exists()]
        if missing:
            failed.append(case_id)
            continue

        out_paths = nnunet_test_paths_for_case(cid, eval_input_dir, len(modalities))
        try:
            for i, mod in enumerate(modalities):
                copy_nifti(in_paths[mod], out_paths[f"channel_{i}"])
            ok_ids.append(case_id)
        except OSError:
            failed.append(case_id)

    return ok_ids, failed


def build_predict_command(
    paths: dict,
    cfg: dict,
    *,
    input_dir: Path,
    output_dir: Path,
    folds: list[int],
    device: str,
    checkpoint: str,
) -> list[str]:
    cmd = [
        find_nnunet_command("nnUNetv2_predict"),
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        str(paths["dataset_id"]),
        "-c",
        str(paths["configuration"]),
        "-tr",
        str(paths["trainer"]),
        "-p",
        str(paths["plans"]),
        "-f",
        *[str(f) for f in folds],
        "-chk",
        checkpoint,
        "-device",
        device,
    ]
    npp = cfg.get("npp")
    if npp is not None:
        cmd.extend(["-npp", str(int(npp))])
    nps = cfg.get("nps")
    if nps is not None:
        cmd.extend(["-nps", str(int(nps))])
    if cfg.get("disable_tta", False):
        cmd.append("--disable_tta")
    if cfg.get("disable_progress_bar", True):
        cmd.append("--disable_progress_bar")
    return cmd


def evaluate_predictions(
    case_ids: list[str],
    *,
    pred_dir: Path,
    ref_masks_dir: Path,
    label_file: str,
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    wt_vals: list[float] = []
    tc_vals: list[float] = []
    et_vals: list[float] = []

    for case_id in case_ids:
        cid = nnunet_case_id(case_id)
        pred_path = pred_dir / label_filename(cid)
        ref_path = output_paths_for_case(
            case_id, Path("_unused"), ref_masks_dir, [], label_file
        )[label_file]
        rec = {
            "case_id": cid,
            "subset": "",
            "status": STATUS_FAIL,
            "reason": "",
            "WT": np.nan,
            "TC": np.nan,
            "ET": np.nan,
            "mean": np.nan,
            "pred_path": str(pred_path),
            "ref_path": str(ref_path),
        }
        if not pred_path.exists():
            rec["reason"] = "missing_prediction"
            records.append(rec)
            continue
        if not ref_path.exists():
            rec["reason"] = "missing_reference"
            records.append(rec)
            continue
        try:
            pred_arr = np.asarray(sitk.GetArrayFromImage(read_nifti(pred_path)), dtype=np.uint8)
            ref_arr = np.asarray(sitk.GetArrayFromImage(read_nifti(ref_path)), dtype=np.uint8)
            scores = dice_regions(pred_arr, ref_arr)
            rec.update(scores)
            rec["mean"] = mean_dice(scores)
            rec["status"] = STATUS_OK
            wt_vals.append(scores["WT"])
            tc_vals.append(scores["TC"])
            et_vals.append(scores["ET"])
        except Exception as exc:  # noqa: BLE001
            rec["reason"] = "metric_error"
            rec["details"] = str(exc)
        records.append(rec)

    summary = {
        "n_cases": len(case_ids),
        "n_ok": len(wt_vals),
        "WT": float(np.mean(wt_vals)) if wt_vals else None,
        "TC": float(np.mean(tc_vals)) if tc_vals else None,
        "ET": float(np.mean(et_vals)) if et_vals else None,
        "mean": float(np.mean([np.mean(wt_vals), np.mean(tc_vals), np.mean(et_vals)]))
        if wt_vals
        else None,
    }
    return records, summary


def _resolve_checkpoint(paths: dict, cfg: dict, folds: list[int]) -> tuple[Path, int]:
    checkpoint_name = str(cfg.get("checkpoint", "checkpoint_final.pth"))
    results_dir = expected_results_dir(
        paths["nnunet_results"],
        dataset_name=paths["dataset_name"],
        trainer=paths["trainer"],
        plans=paths["plans"],
        configuration=paths["configuration"],
    )
    for fold in folds:
        ckpt = fold_checkpoint_path(results_dir, fold, checkpoint_name)
        if ckpt.exists():
            return ckpt, fold
        status, reason, details = verify_fold_checkpoint(
            {
                "nnunet_results": paths["nnunet_results"],
                "dataset_name": paths["dataset_name"],
                "trainer": paths["trainer"],
                "plans": paths["plans"],
                "configuration": paths["configuration"],
            },
            fold,
        )
        if status == TRAIN_OK and details.get("checkpoint"):
            return Path(details["checkpoint"]), fold
    raise FileNotFoundError(
        f"No se encontró checkpoint '{checkpoint_name}' para folds {folds} en {results_dir}"
    )


def run_eval(config: dict, *, yes: bool = False) -> pd.DataFrame:
    """Inferencia nnU-Net + métricas WT/TC/ET en val/test hold-out."""
    paths = eval_paths(config)
    cfg = config.get("eval", {})
    s1 = config.get("stage_1", {})
    s3 = config.get("stage_3", {})
    s4 = config.get("stage_4", {})
    nn_cfg = config.get("nnunet", {})

    modalities: list[str] = list(s3.get("modalities") or s1.get("modalities", ["t1", "t1ce", "t2", "flair"]))
    label_file: str = s3.get("label_file", "seg")
    subsets: list[str] = list(cfg.get("subsets", ["val", "test"]))
    folds = parse_folds(cfg.get("folds", s4.get("folds", nn_cfg.get("fold", 0))))
    device = detect_device(str(cfg.get("device", s4.get("device", "auto"))))
    checkpoint_name = str(cfg.get("checkpoint", "checkpoint_final.pth"))

    if not paths["split_json"].exists():
        raise FileNotFoundError(f"Falta split.json: {paths['split_json']}")

    subset_cases = load_eval_subsets(paths["split_json"], subsets)
    checkpoint_path, fold_used = _resolve_checkpoint(paths, cfg, folds)

    ensure_dir(paths["output_dir"])
    ensure_dir(paths["output_log"].parent)

    env = apply_nnunet_env(
        nnunet_raw=paths["nnunet_raw"],
        nnunet_preprocessed=paths["nnunet_preprocessed"],
        nnunet_results=paths["nnunet_results"],
    )

    print("[*] Eval: inferencia + Dice WT/TC/ET (hold-out val/test)")
    print(f"    split.json:   {paths['split_json']}")
    print(f"    checkpoint:   {checkpoint_path}")
    print(f"    fold:         {fold_used}")
    print(f"    device:       {device}")
    print(f"    subsets:      {', '.join(f'{k}({len(v)})' for k, v in subset_cases.items())}")
    print(f"    output:       {paths['output_dir']}")

    if not yes:
        answer = input("Continuar? [y/N]: ").strip().lower()
        if answer not in {"y", "yes", "s", "si"}:
            print("[-] Cancelado por el usuario.")
            return pd.DataFrame()

    all_records: list[dict] = []
    subset_summaries: dict[str, dict] = {}

    for subset_name, case_ids in subset_cases.items():
        print(f"\n[*] Subset: {subset_name} ({len(case_ids)} casos)")
        eval_input = paths["output_dir"] / f"input_{subset_name}"
        pred_output = paths["output_dir"] / f"predictions_{subset_name}"

        if eval_input.exists() and cfg.get("clean_input", True):
            shutil.rmtree(eval_input)
        if pred_output.exists() and cfg.get("clean_predictions", True):
            shutil.rmtree(pred_output)
        ensure_dir(eval_input)
        ensure_dir(pred_output)

        ok_ids, failed_prep = prepare_eval_input(
            case_ids,
            input_volumes_dir=paths["input_volumes_dir"],
            eval_input_dir=eval_input,
            modalities=modalities,
        )
        if failed_prep:
            print(f"    [!] {len(failed_prep)} casos sin volúmenes Stage 1: {failed_prep[:5]}...")
        print(f"    input preparado: {len(ok_ids)}/{len(case_ids)}")

        if not ok_ids:
            print(f"    [-] Sin casos válidos para {subset_name}, omitiendo inferencia.")
            continue

        t0 = time.time()
        cmd = build_predict_command(
            paths,
            cfg,
            input_dir=eval_input,
            output_dir=pred_output,
            folds=[fold_used],
            device=device,
            checkpoint=checkpoint_name,
        )
        proc = run_command(cmd, env=env)
        elapsed_pred = round(time.time() - t0, 1)
        print(f"    inferencia: {elapsed_pred}s (exit={proc.returncode})")

        if proc.returncode != 0:
            raise RuntimeError(
                f"nnUNetv2_predict falló para subset {subset_name} (exit {proc.returncode})"
            )

        records, summary = evaluate_predictions(
            ok_ids,
            pred_dir=pred_output,
            ref_masks_dir=paths["input_masks_dir"],
            label_file=label_file,
        )
        for rec in records:
            rec["subset"] = subset_name
            rec["elapsed_pred_sec"] = elapsed_pred
            rec["checkpoint"] = str(checkpoint_path)
            rec["fold"] = fold_used
        all_records.extend(records)

        summary["subset"] = subset_name
        summary["elapsed_pred_sec"] = elapsed_pred
        summary["checkpoint"] = str(checkpoint_path)
        summary["fold"] = fold_used
        subset_summaries[subset_name] = summary

        if summary.get("mean") is not None:
            print(
                f"    Dice {subset_name}: WT={summary['WT']:.3f} "
                f"TC={summary['TC']:.3f} ET={summary['ET']:.3f} "
                f"mean={summary['mean']:.3f}"
            )

    log_df = pd.DataFrame(all_records)
    log_df.to_csv(paths["output_log"], index=False)

    summary_doc = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "fold": fold_used,
        "configuration": paths["configuration"],
        "subsets": subset_summaries,
        "per_case": all_records,
    }
    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n[*] Log: {paths['output_log']}")
    print(f"[*] Resumen: {paths['summary_json']}")
    n_ok = int((log_df["status"] == STATUS_OK).sum()) if not log_df.empty else 0
    n_fail = len(log_df) - n_ok
    print(f"[*] Casos evaluados: {n_ok} OK, {n_fail} FAILED")
    return log_df
