#!/usr/bin/env python3
"""Auditoría post-hoc del Stage 2 (split train/test + bundles + test GT)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline_io import load_config
from src.data.stage_2 import audit_split_integrity, load_approved_case_ids, stage2_paths


def audit_stage_2(config: dict) -> Path:
    paths = stage2_paths(config)
    report_path = paths["audit_report"]
    log_path = paths["output_log"]
    manifest_path = paths["split_manifest"]

    case_ids = load_approved_case_ids(paths["input_log"])
    max_cases = config.get("stage_2", {}).get("max_cases")
    if max_cases is not None:
        case_ids = case_ids[: int(max_cases)]

    rows, _ = audit_split_integrity(config)

    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print("[*] Auditoría Stage 2 (Split Manager)")
    print(f"    manifest:  {manifest_path}")
    print(f"    splits:    {paths['splits_dir']}")
    print(f"    test GT:   {paths['test_ground_truth_dir']}")
    print(f"    casos ref: {len(case_ids)} (Stage 1 OK)")

    for row in rows:
        flag = "+" if row["ok"] else "-"
        print(f"  [{flag}] {row['check']}: {row['detail']}")

    log_summary = "No se encontró processing_log.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        n_test = int((log_df.get("split") == "test").sum()) if "split" in log_df.columns else 0
        log_summary = f"processing_log.csv: {len(log_df)} filas; test={n_test}"

    all_ok = all(r["ok"] for r in rows)
    counts = manifest.get("counts") or {}

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Reporte de auditoría Stage 2",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Resumen",
        "",
        f"- Casos referencia (Stage 1 OK): {len(case_ids)}",
        f"- Manifest: `{manifest_path}`",
        f"- counts: train={counts.get('train', '?')}, test={counts.get('test', '?')}",
        f"- Checks integridad: {'OK' if all_ok else 'FAILED'}",
        f"- {log_summary}",
        "",
        "## Criterios",
        "",
        "- Cero solapamiento entre train y test.",
        "- Suma train+test = total casos procesados.",
        "- Train: imagesTr + labelsTr emparejados; Test: imagesTs + labelsTs emparejados.",
        "- `test_ground_truth/` coincide exactamente con la lista `test` del manifest.",
        "- Test: labels también en test/labelsTs; test_ground_truth es copia plana para eval Stage 4.",
        "",
        "## Checks",
        "",
        "| check | ok | detail |",
        "|-------|----|--------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['check']} | {'yes' if row['ok'] else 'no'} | {row['detail']} |"
        )

    lines.extend(
        [
            "",
            "## Rutas",
            "",
            f"- Manifest: `{manifest_path}`",
            f"- Splits: `{paths['splits_dir']}`",
            f"- Test GT: `{paths['test_ground_truth_dir']}`",
            f"- Log: `{log_path}`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[*] Reporte: {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditar salida Stage 2 (split)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    audit_stage_2(load_config(args.config))


if __name__ == "__main__":
    main()
