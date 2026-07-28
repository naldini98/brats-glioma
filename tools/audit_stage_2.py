#!/usr/bin/env python3
"""Auditoría post-hoc del Stage 2 (formato nnU-Net v2)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline_io import load_config
from src.data.stage_2 import STATUS_FAIL, STATUS_OK, stage2_paths, verify_stage_2_output


def audit_stage_2(config: dict) -> Path:
    paths = stage2_paths(config)
    cfg = config["stage_2"]
    modalities = list(cfg.get("modalities") or config["stage_1"]["modalities"])

    images_tr = paths["images_tr"]
    labels_tr = paths["labels_tr"]
    report_path = paths["audit_report"]
    log_path = paths["output_log"]
    dataset_json = paths["dataset_json"]

    case_ids: list[str] = []
    if labels_tr.exists():
        case_ids = sorted(
            p.name.replace(".nii.gz", "").replace(".nii", "")
            for p in labels_tr.glob("*.nii*")
            if not p.name.startswith("._")
        )

    rows: list[dict] = []
    print("[*] Auditoría Stage 2")
    print(f"    dataset:  {paths['dataset_folder']}")
    print(f"    imagesTr: {images_tr}")
    print(f"    labelsTr: {labels_tr}")
    print(f"    casos en disco: {len(case_ids)}")

    for case_id in case_ids:
        status, reason, details = verify_stage_2_output(
            case_id, images_tr, labels_tr, modalities
        )
        geom = details.get("geometry") or {}
        size = geom.get("size")
        size_str = "x".join(str(s) for s in size) if size else ""
        rows.append(
            {
                "case_id": case_id,
                "status": status,
                "reason": reason,
                "size": size_str,
                "details": details.get("details", "") if status != STATUS_OK else "OK",
            }
        )
        flag = "+" if status == STATUS_OK else "-"
        print(f"  [{flag}] {case_id}: {status}" + (f" ({reason})" if reason else ""))

    df = pd.DataFrame(rows)
    n_ok = int((df["status"] == STATUS_OK).sum()) if not df.empty else 0
    n_fail = len(df) - n_ok

    log_summary = "No se encontró conversion_log.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        log_ok = int((log_df["status"] == STATUS_OK).sum())
        log_fail = len(log_df) - log_ok
        log_summary = f"conversion_log.csv: {log_ok} OK, {log_fail} FAILED (total {len(log_df)})"

    json_summary = "dataset.json: ausente"
    if dataset_json.exists():
        meta = json.loads(dataset_json.read_text(encoding="utf-8"))
        json_summary = (
            f"dataset.json: numTraining={meta.get('numTraining')}, "
            f"channels={meta.get('channel_names')}, labels={meta.get('labels')}"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Reporte de auditoría Stage 2",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Resumen",
        "",
        f"- Casos auditados en disco: {len(case_ids)}",
        f"- {STATUS_OK}: {n_ok}",
        f"- {STATUS_FAIL}: {n_fail}",
        f"- {log_summary}",
        f"- {json_summary}",
        "",
        "## Criterios",
        "",
        f"- {len(modalities)} canales por caso (`_0000`..`_{len(modalities)-1:04d}`) + label.",
        f"- Orden de canales: {modalities}.",
        "- Mismo size, spacing, origin y direction (mellizos).",
        "- MRI float; seg uint8 con clases Decathlon {0,1,2,3} (sin remap).",
        "- dataset.json presente y coherente con numTraining.",
        "",
        "## Detalle por caso",
        "",
        "| case_id | status | reason | size | details |",
        "|---------|--------|--------|------|---------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['status']} | {row['reason'] or '-'} | "
            f"{row['size'] or '-'} | {row['details']} |"
        )
    if not rows:
        lines.append("| (sin casos) | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Rutas",
            "",
            f"- Dataset: `{paths['dataset_folder']}`",
            f"- imagesTr: `{images_tr}`",
            f"- labelsTr: `{labels_tr}`",
            f"- dataset.json: `{dataset_json}`",
            f"- Log: `{log_path}`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[*] Reporte: {report_path}")
    print(f"[*] Resumen: {n_ok} OK, {n_fail} FAILED")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditar salida Stage 2")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    audit_stage_2(load_config(args.config))


if __name__ == "__main__":
    main()
