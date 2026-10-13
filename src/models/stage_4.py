"""Stage 4: Plan/preprocess + entrenamiento nnU-Net v2.

Entrada: output/nnunet_raw/Dataset101_BraTS (Stage 3, solo train split).
Pasos:
  1) nnUNetv2_plan_and_preprocess
  2) nnUNetv2_train

Salidas:
  output/nnunet_preprocessed/Dataset101_BraTS/
  output/nnunet_results/Dataset101_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.dicom_geometry import ensure_dir
from src.data.nnunet_convert import dataset_folder_name
from src.data.stage_0 import project_root_from_config
from src.models.nnunet_env import (
    apply_nnunet_env,
    detect_device,
    expected_results_dir,
    find_nnunet_command,
    fold_checkpoint_path,
    parse_folds,
    plans_json_path,
    resolve_under_root,
    run_command,
)

STATUS_OK = "SUCCESS_TRAINED"
STATUS_PLANNED = "SUCCESS_PLANNED"
STATUS_FAIL = "FAILED"


def stage4_paths(config: dict) -> dict:
    root = project_root_from_config(config)
    cfg = config.get("stage_4", {})
    nn_cfg = config.get("nnunet", {})
    s3 = config.get("stage_3", {})

    def _cfg_path(key: str, default: str) -> Path:
        value = cfg.get(key) or config.get("paths", {}).get(key) or s3.get(key) or default
        return resolve_under_root(root, value)

    nnunet_raw = _cfg_path("nnunet_raw", s3.get("nnunet_raw", "output/nnunet_raw"))
    nnunet_preprocessed = _cfg_path("nnunet_preprocessed", "output/nnunet_preprocessed")
    nnunet_results = _cfg_path("nnunet_results", "output/nnunet_results")

    dataset_id = int(nn_cfg.get("dataset_id", 101))
    dataset_name = str(nn_cfg.get("dataset_name", "Dataset101_BraTS"))
    folder_name = dataset_folder_name(dataset_id, dataset_name)

    output_log = resolve_under_root(root, cfg.get("output_log", "output/stage_4/training_log.csv"))
    audit_report = resolve_under_root(
        root, cfg.get("audit_report", "output/stage_4/reporte_auditoria_stage_4.md")
    )

    return {
        "root": root,
        "nnunet_raw": nnunet_raw,
        "nnunet_preprocessed": nnunet_preprocessed,
        "nnunet_results": nnunet_results,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "dataset_folder": nnunet_raw / folder_name,
        "dataset_json": nnunet_raw / folder_name / "dataset.json",
        "images_tr": nnunet_raw / folder_name / "imagesTr",
        "labels_tr": nnunet_raw / folder_name / "labelsTr",
        "output_log": output_log,
        "audit_report": audit_report,
        "configuration": str(cfg.get("configuration") or nn_cfg.get("configuration", "3d_fullres")),
        "trainer": str(cfg.get("trainer", "nnUNetTrainer")),
        "plans": str(cfg.get("plans", "nnUNetPlans")),
    }


def verify_stage3_dataset(paths: dict) -> tuple[bool, str]:
    """Comprueba que Stage 3 dejó un dataset usable."""
    if not paths["dataset_json"].exists():
        return False, f"Falta dataset.json: {paths['dataset_json']}"
    if not paths["images_tr"].exists():
        return False, f"Falta imagesTr: {paths['images_tr']}"
    if not paths["labels_tr"].exists():
        return False, f"Falta labelsTr: {paths['labels_tr']}"
    n_labels = len(list(paths["labels_tr"].glob("*.nii*")))
    if n_labels < 1:
        return False, "labelsTr vacío"
    return True, f"{n_labels} casos en labelsTr"


def verify_plans(paths: dict) -> tuple[str, str, dict]:
    plans = plans_json_path(paths["nnunet_preprocessed"], paths["dataset_name"], paths["plans"])
    if plans.exists():
        return STATUS_PLANNED, "", {"plans_json": str(plans)}
    return STATUS_FAIL, "missing_plans", {"expected": str(plans)}


def verify_fold_checkpoint(paths: dict, fold: int) -> tuple[str, str, dict]:
    results_dir = expected_results_dir(
        paths["nnunet_results"],
        dataset_name=paths["dataset_name"],
        trainer=paths["trainer"],
        plans=paths["plans"],
        configuration=paths["configuration"],
    )
    final_ckpt = fold_checkpoint_path(results_dir, fold, "checkpoint_final.pth")
    best_ckpt = fold_checkpoint_path(results_dir, fold, "checkpoint_best.pth")
    latest_ckpt = fold_checkpoint_path(results_dir, fold, "checkpoint_latest.pth")

    if final_ckpt.exists():
        return STATUS_OK, "", {"checkpoint": str(final_ckpt), "fold": fold}
    if best_ckpt.exists():
        return STATUS_OK, "checkpoint_best_only", {"checkpoint": str(best_ckpt), "fold": fold}
    if latest_ckpt.exists():
        return STATUS_FAIL, "training_incomplete", {"checkpoint": str(latest_ckpt), "fold": fold}
    return STATUS_FAIL, "missing_checkpoint", {"expected": str(final_ckpt), "fold": fold}


def build_plan_command(paths: dict, cfg: dict) -> list[str]:
    cmd = [
        find_nnunet_command("nnUNetv2_plan_and_preprocess"),
        "-d",
        str(paths["dataset_id"]),
    ]
    if cfg.get("verify_dataset_integrity", True):
        cmd.append("--verify_dataset_integrity")
    if cfg.get("clean_fingerprint", False):
        cmd.append("--clean")
    if cfg.get("no_preprocess", False):
        cmd.append("--no_pp")

    np_plan = cfg.get("np")
    if np_plan is not None:
        cmd.extend(["-np", str(int(np_plan))])
    npfp = cfg.get("np_fingerprint")
    if npfp is not None:
        cmd.extend(["-npfp", str(int(npfp))])

    configurations = cfg.get("configurations")
    if configurations:
        cmd.append("-c")
        cmd.extend([str(c) for c in configurations])
    elif paths.get("configuration"):
        cmd.extend(["-c", str(paths["configuration"])])

    return cmd


def build_train_command(
    paths: dict,
    cfg: dict,
    *,
    fold: int,
    device: str,
) -> list[str]:
    cmd = [
        find_nnunet_command("nnUNetv2_train"),
        str(paths["dataset_id"]),
        str(paths["configuration"]),
        str(fold),
        "-tr",
        str(paths["trainer"]),
        "-p",
        str(paths["plans"]),
        "-device",
        device,
    ]
    if cfg.get("continue_training", False):
        cmd.append("--c")
    if cfg.get("npz", False):
        cmd.append("--npz")
    if cfg.get("only_validation", False):
        cmd.append("--val")
    num_gpus = cfg.get("num_gpus")
    if num_gpus is not None:
        cmd.extend(["-num_gpus", str(int(num_gpus))])
    return cmd


def run_stage_4(config: dict, *, yes: bool = False) -> pd.DataFrame:
    """Orquesta Stage 4: plan/preprocess + train."""
    paths = stage4_paths(config)
    cfg = config.get("stage_4", {})
    nn_cfg = config.get("nnunet", {})

    folds = parse_folds(cfg.get("folds", nn_cfg.get("fold", 0)))
    device = detect_device(str(cfg.get("device", "auto")))
    run_plan = bool(cfg.get("run_plan_and_preprocess", True))
    run_train = bool(cfg.get("run_train", True))

    ensure_dir(paths["nnunet_raw"])
    ensure_dir(paths["nnunet_preprocessed"])
    ensure_dir(paths["nnunet_results"])
    ensure_dir(paths["output_log"].parent)

    ok_ds, ds_msg = verify_stage3_dataset(paths)
    if not ok_ds:
        raise FileNotFoundError(f"Stage 3 incompleto: {ds_msg}")

    env = apply_nnunet_env(
        nnunet_raw=paths["nnunet_raw"],
        nnunet_preprocessed=paths["nnunet_preprocessed"],
        nnunet_results=paths["nnunet_results"],
    )

    print("[*] Stage 4: nnU-Net v2 plan/preprocess + train")
    print(f"    dataset:       {paths['dataset_folder']}")
    print(f"    raw:           {paths['nnunet_raw']}")
    print(f"    preprocessed:  {paths['nnunet_preprocessed']}")
    print(f"    results:       {paths['nnunet_results']}")
    print(f"    dataset_id:    {paths['dataset_id']}")
    print(f"    configuration: {paths['configuration']}")
    print(f"    trainer:       {paths['trainer']}")
    print(f"    plans:         {paths['plans']}")
    print(f"    folds:         {folds}")
    print(f"    device:        {device}")
    print(f"    plan+pp:       {run_plan}")
    print(f"    train:         {run_train}")
    print(f"    Stage3 check:  {ds_msg}")

    if device == "cpu":
        print(
            "[!] ADVERTENCIA: entrenando en CPU. BraTS 3d_fullres es muy lento sin GPU "
            "(días/semanas). Preferí CUDA."
        )

    if not yes:
        answer = input("Continuar? [y/N]: ").strip().lower()
        if answer not in {"y", "yes", "s", "si"}:
            print("[-] Cancelado por el usuario.")
            return pd.DataFrame()

    records: list[dict] = []

    if run_plan:
        step = "plan_and_preprocess"
        print(f"[*] Paso: {step}")
        t0 = time.time()
        cmd = build_plan_command(paths, cfg)
        proc = run_command(cmd, env=env)
        elapsed = round(time.time() - t0, 1)
        status, reason, details = verify_plans(paths)
        if proc.returncode != 0:
            status, reason = STATUS_FAIL, f"exit_code_{proc.returncode}"
        records.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "step": step,
                "configuration": paths["configuration"],
                "fold": "",
                "status": status if status != STATUS_OK else STATUS_PLANNED,
                "reason": reason,
                "details": str(details),
                "elapsed_sec": elapsed,
                "device": device,
                "checkpoint": details.get("plans_json", ""),
            }
        )
        flag = "+" if status != STATUS_FAIL else "-"
        print(f"  [{flag}] {step}: {records[-1]['status']} ({elapsed}s)")
        if status == STATUS_FAIL:
            _write_log(records, paths["output_log"])
            print(f"[*] Log: {paths['output_log']}")
            print("[!] Stage 4 detenido: falló plan_and_preprocess")
            return pd.DataFrame(records)

    if run_train:
        for fold in folds:
            step = f"train_fold_{fold}"
            print(f"[*] Paso: {step}")
            t0 = time.time()
            cmd = build_train_command(paths, cfg, fold=fold, device=device)
            proc = run_command(cmd, env=env)
            elapsed = round(time.time() - t0, 1)
            status, reason, details = verify_fold_checkpoint(paths, fold)
            if proc.returncode != 0 and status == STATUS_OK:
                reason = f"exit_code_{proc.returncode}_with_checkpoint"
            elif proc.returncode != 0:
                status, reason = STATUS_FAIL, f"exit_code_{proc.returncode}"
            records.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "step": step,
                    "configuration": paths["configuration"],
                    "fold": fold,
                    "status": status,
                    "reason": reason,
                    "details": str(details),
                    "elapsed_sec": elapsed,
                    "device": device,
                    "checkpoint": details.get("checkpoint", details.get("expected", "")),
                }
            )
            flag = "+" if status == STATUS_OK else "-"
            print(f"  [{flag}] {step}: {status} {reason} ({elapsed}s)".rstrip())
            if status == STATUS_FAIL and not cfg.get("continue_on_fold_fail", False):
                break

    log_df = _write_log(records, paths["output_log"])
    n_ok = int(log_df["status"].isin([STATUS_OK, STATUS_PLANNED]).sum()) if not log_df.empty else 0
    n_fail = int((log_df["status"] == STATUS_FAIL).sum()) if not log_df.empty else 0
    print(f"[*] Log: {paths['output_log']}")
    print(f"[*] Resumen: {n_ok} OK/PLANNED, {n_fail} FAILED")
    return log_df


def _write_log(records: list[dict], path: Path) -> pd.DataFrame:
    ensure_dir(path.parent)
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    return df
