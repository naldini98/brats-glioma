#!/usr/bin/env python3
"""Auditoría post-hoc del Stage 4 (planes + checkpoints nnU-Net)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline_io import load_config
from src.models.nnunet_env import expected_results_dir, parse_folds
from src.models.stage_4 import (
    STATUS_FAIL,
    STATUS_OK,
    STATUS_PLANNED,
    stage4_paths,
    verify_fold_checkpoint,
    verify_plans,
    verify_stage3_dataset,
)


def audit_stage_4(config: dict) -> Path:
    paths = stage4_paths(config)
    cfg = config.get("stage_4", {})
    nn_cfg = config.get("nnunet", {})
    folds = parse_folds(cfg.get("folds", nn_cfg.get("fold", 0)))
    report_path = paths["audit_report"]
    log_path = paths["output_log"]

    print("[*] Auditoría Stage 4")
    print(f"    raw:          {paths['nnunet_raw']}")
    print(f"    preprocessed: {paths['nnunet_preprocessed']}")
    print(f"    results:      {paths['nnunet_results']}")
    print(f"    folds:        {folds}")

    rows: list[dict] = []

    ok_ds, ds_msg = verify_stage3_dataset(paths)
    rows.append(
        {
            "step": "stage3_dataset",
            "status": STATUS_PLANNED if ok_ds else STATUS_FAIL,
            "reason": "" if ok_ds else "missing_dataset",
            "details": ds_msg,
        }
    )
    flag = "+" if ok_ds else "-"
    print(f"  [{flag}] stage3_dataset: {ds_msg}")

    status, reason, details = verify_plans(paths)
    rows.append(
        {
            "step": "plans",
            "status": status,
            "reason": reason,
            "details": details.get("plans_json") or details.get("expected", ""),
        }
    )
    flag = "+" if status != STATUS_FAIL else "-"
    print(f"  [{flag}] plans: {status} {reason}".rstrip())

    for fold in folds:
        st, rs, det = verify_fold_checkpoint(paths, fold)
        rows.append(
            {
                "step": f"fold_{fold}",
                "status": st,
                "reason": rs,
                "details": det.get("checkpoint") or det.get("expected", ""),
            }
        )
        flag = "+" if st == STATUS_OK else "-"
        print(f"  [{flag}] fold_{fold}: {st} {rs}".rstrip())

    results_dir = expected_results_dir(
        paths["nnunet_results"],
        dataset_name=paths["dataset_name"],
        trainer=paths["trainer"],
        plans=paths["plans"],
        configuration=paths["configuration"],
    )

    log_summary = "No se encontró training_log.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        log_summary = (
            f"training_log.csv: {len(log_df)} filas; "
            f"statuses={log_df['status'].value_counts().to_dict() if 'status' in log_df else {}}"
        )

    n_ok = sum(1 for r in rows if r["status"] in {STATUS_OK, STATUS_PLANNED})
    n_fail = sum(1 for r in rows if r["status"] == STATUS_FAIL)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Reporte de auditoría Stage 4",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Resumen",
        "",
        f"- Checks OK/PLANNED: {n_ok}",
        f"- Checks FAILED: {n_fail}",
        f"- {log_summary}",
        "",
        "## Criterios",
        "",
        "- Dataset Stage 3 presente (imagesTr/labelsTr/dataset.json).",
        f"- Plans: `{{nnUNet_preprocessed}}/{paths['dataset_name']}/{paths['plans']}.json`.",
        f"- Checkpoint por fold: `{results_dir}/fold_X/checkpoint_final.pth`.",
        f"- Configuración: {paths['configuration']} | trainer: {paths['trainer']}.",
        "",
        "## Detalle",
        "",
        "| step | status | reason | details |",
        "|------|--------|--------|---------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | {row['status']} | {row['reason'] or '-'} | {row['details']} |"
        )
    lines.extend(
        [
            "",
            "## Rutas",
            "",
            f"- Dataset raw: `{paths['dataset_folder']}`",
            f"- Preprocessed: `{paths['nnunet_preprocessed']}`",
            f"- Results: `{results_dir}`",
            f"- Log: `{log_path}`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[*] Reporte: {report_path}")
    print(f"[*] Resumen: {n_ok} OK/PLANNED, {n_fail} FAILED")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditar salida Stage 4")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    audit_stage_4(load_config(args.config))


if __name__ == "__main__":
    main()
