#!/usr/bin/env python3
"""Auditoría post-hoc del Stage 1 (RAS + 1mm + normalización)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline_io import load_config
from src.data.stage_1 import STATUS_FAIL, STATUS_OK, stage1_paths, verify_stage_1_output


def audit_stage_1(config: dict) -> Path:
    paths = stage1_paths(config)
    cfg = config["stage_1"]
    modalities = list(cfg["modalities"])
    label_file = cfg.get("label_file", "seg")
    target_spacing = tuple(float(x) for x in cfg["target_spacing"])

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
    print("[*] Auditoría Stage 1")
    print(f"    volumes: {volumes_dir}")
    print(f"    masks:   {masks_dir}")
    print(f"    casos en disco: {len(case_ids)}")

    for case_id in case_ids:
        status, reason, details = verify_stage_1_output(
            case_id,
            volumes_dir,
            masks_dir,
            modalities,
            label_file,
            target_spacing=target_spacing,
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
                "spacing": "x".join(str(round(s, 4)) for s in details.get("spacing", ())),
                "details": details.get("details", "") if status != STATUS_OK else "OK",
            }
        )
        flag = "+" if status == STATUS_OK else "-"
        print(f"  [{flag}] {case_id}: {status}" + (f" ({reason})" if reason else ""))

    df = pd.DataFrame(rows)
    n_ok = int((df["status"] == STATUS_OK).sum()) if not df.empty else 0
    n_fail = len(df) - n_ok

    log_summary = "No se encontró preprocessing_log.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        log_ok = int((log_df["status"] == STATUS_OK).sum())
        log_fail = len(log_df) - log_ok
        log_summary = f"preprocessing_log.csv: {log_ok} OK, {log_fail} FAILED (total {len(log_df)})"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Reporte de auditoría Stage 1",
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
        "- 5 archivos .nii.gz por caso.",
        "- Mismo size, spacing, origin y direction (mellizos).",
        f"- Spacing objetivo: {target_spacing} mm.",
        "- MRI float32 normalizada; seg uint8 con clases {0,1,2,3}.",
        "- Normalización: clip percentil + z-score por modalidad/caso (sin [0,1]).",
        "",
        "## Detalle por caso",
        "",
        "| case_id | status | reason | shapes | spacing | details |",
        "|---------|--------|--------|--------|---------|---------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['status']} | {row['reason'] or '-'} | "
            f"{row['shapes'] or '-'} | {row['spacing'] or '-'} | {row['details']} |"
        )
    if not rows:
        lines.append("| (sin casos) | - | - | - | - | - |")

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
    parser = argparse.ArgumentParser(description="Auditar salida Stage 1")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    audit_stage_1(load_config(args.config))


if __name__ == "__main__":
    main()
