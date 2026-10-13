#!/usr/bin/env python3
"""Auditoría post-hoc del Stage 2 (split 70/15/15)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline_io import load_config
from src.data.stage_2 import STATUS_FAIL, STATUS_OK, load_split_json, stage2_paths, verify_split


def audit_stage_2(config: dict) -> Path:
    paths = stage2_paths(config)
    report_path = paths["audit_report"]
    log_path = paths["output_log"]
    split_path = paths["split_json"]

    print("[*] Auditoría Stage 2 (split)")
    print(f"    split.json: {split_path}")
    print(f"    log:        {log_path}")

    rows: list[dict] = []
    if not split_path.exists():
        rows.append(
            {
                "check": "split_json",
                "status": STATUS_FAIL,
                "reason": "missing_file",
                "details": str(split_path),
            }
        )
        print(f"  [-] split.json ausente")
    else:
        payload = load_split_json(split_path)
        status, reason, details = verify_split(payload)
        rows.append(
            {
                "check": "split_integrity",
                "status": status,
                "reason": reason,
                "details": str(details),
            }
        )
        flag = "+" if status == STATUS_OK else "-"
        counts = details.get("counts", {})
        print(
            f"  [{flag}] split: train={counts.get('train', '?')} "
            f"val={counts.get('val', '?')} test={counts.get('test', '?')} seed={details.get('seed')}"
        )

    log_summary = "No se encontró split_log.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        log_summary = f"split_log.csv: {log_df.iloc[0].to_dict() if not log_df.empty else {}}"

    n_ok = sum(1 for r in rows if r["status"] == STATUS_OK)
    n_fail = len(rows) - n_ok

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Reporte de auditoría Stage 2",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Resumen",
        "",
        f"- {STATUS_OK}: {n_ok}",
        f"- {STATUS_FAIL}: {n_fail}",
        f"- {log_summary}",
        "",
        "## Criterios",
        "",
        "- split.json presente con listas train/val/test sin solapamiento.",
        "- Ratios ~70/15/15 con seed fija reproducible.",
        "- Todos los case IDs provienen de Stage 1 OK.",
        "",
        "## Detalle",
        "",
        "| check | status | reason | details |",
        "|-------|--------|--------|---------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['check']} | {row['status']} | {row['reason'] or '-'} | {row['details']} |"
        )

    if split_path.exists():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "## Conteos",
                "",
                f"- total: {payload.get('total')}",
                f"- train: {payload.get('counts', {}).get('train', len(payload.get('train', [])))}",
                f"- val: {payload.get('counts', {}).get('val', len(payload.get('val', [])))}",
                f"- test: {payload.get('counts', {}).get('test', len(payload.get('test', [])))}",
                f"- seed: {payload.get('seed')}",
                "",
                "## Rutas",
                "",
                f"- split.json: `{split_path}`",
                f"- Log: `{log_path}`",
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[*] Reporte: {report_path}")
    print(f"[*] Resumen: {n_ok} OK, {n_fail} FAILED")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditar salida Stage 2 (split)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    audit_stage_2(load_config(args.config))


if __name__ == "__main__":
    main()
