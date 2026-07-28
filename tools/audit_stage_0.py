#!/usr/bin/env python3
"""Auditoría post-hoc del Stage 0 (Decathlon → NIfTI mellizos)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline_io import load_config
from src.data.stage_0 import STATUS_FAIL, STATUS_OK, stage0_paths, verify_stage_0_output


def audit_stage_0(config: dict) -> Path:
    paths = stage0_paths(config)
    cfg = config["stage_0"]
    modalities = list(cfg["modalities"])
    label_file = cfg.get("label_file", "seg")

    volumes_dir = paths["volumes_dir"]
    masks_dir = paths["masks_dir"]
    report_path = paths["audit_report"]
    log_path = paths["output_log"]

    case_ids: list[str] = []
    if volumes_dir.exists():
        case_ids = sorted(
            p.name.replace("volume_", "", 1)
            for p in volumes_dir.iterdir()
            if p.is_dir() and p.name.startswith("volume_")
        )

    rows: list[dict] = []
    print("[*] Auditoría Stage 0")
    print(f"    volumes: {volumes_dir}")
    print(f"    masks:   {masks_dir}")
    print(f"    casos en disco: {len(case_ids)}")

    for case_id in case_ids:
        status, reason, details = verify_stage_0_output(
            case_id, volumes_dir, masks_dir, modalities, label_file
        )
        shapes = details.get("shapes", {})
        shape_str = ""
        if shapes and modalities[0] in shapes:
            x, y, z = shapes[modalities[0]]
            shape_str = f"{z}x{y}x{x}"
        rows.append(
            {
                "case_id": case_id,
                "status": status,
                "reason": reason,
                "shapes": shape_str,
                "details": details.get("details", "") if status != STATUS_OK else "OK",
            }
        )
        flag = "+" if status == STATUS_OK else "-"
        print(f"  [{flag}] {case_id}: {status}" + (f" ({reason})" if reason else ""))

    df = pd.DataFrame(rows)
    n_ok = int((df["status"] == STATUS_OK).sum()) if not df.empty else 0
    n_fail = len(df) - n_ok

    log_summary = "No se encontró processing_log.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        log_ok = int((log_df["status"] == STATUS_OK).sum())
        log_fail = len(log_df) - log_ok
        log_summary = f"processing_log.csv: {log_ok} OK, {log_fail} FAILED (total {len(log_df)})"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Reporte de auditoría Stage 0",
        "",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Resumen",
        "",
        f"- Casos auditados en disco: {len(case_ids)}",
        f"- {STATUS_OK}: {n_ok}",
        f"- {STATUS_FAIL}: {n_fail}",
        f"- {log_summary}",
        "",
        "## Criterios",
        "",
        "- Existen 5 archivos .nii.gz (t1, t1ce, t2, flair, seg).",
        "- Mismo size en las 5 imágenes.",
        "- Mismos Spacing, Origin y Direction (geometría nativa Decathlon).",
        "- Máscara seg en sitkUInt8; clases Decathlon {0,1,2,3}.",
        "",
        "## Detalle por caso",
        "",
        "| case_id | status | reason | shapes | details |",
        "|---------|--------|--------|--------|---------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['status']} | {row['reason'] or '-'} | "
            f"{row['shapes'] or '-'} | {row['details']} |"
        )
    if not rows:
        lines.append("| (sin casos) | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Rutas",
            "",
            f"- Volúmenes: `{volumes_dir}`",
            f"- Máscaras: `{masks_dir}`",
            f"- Log: `{log_path}`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[*] Reporte: {report_path}")
    print(f"[*] Resumen: {n_ok} OK, {n_fail} FAILED")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditar salida Stage 0")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    audit_stage_0(load_config(args.config))


if __name__ == "__main__":
    main()
