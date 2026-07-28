#!/usr/bin/env python3
"""Auditoría de data/00_raw — wrapper CLI del Stage 0."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline_io import load_config, resolve_path
from src.data.validation import audit_case, scan_raw_cases


def audit_raw(config: dict, base: Path | None = None) -> None:
    root = base or Path.cwd()
    raw_dir = resolve_path(config, "raw", root)
    ingest_cfg = config["ingest"]
    modalities = ingest_cfg["modalities"]
    label_file = ingest_cfg["label_file"]

    if not raw_dir.exists():
        print(f"[-] Directorio raw no encontrado: {raw_dir}")
        return

    print(f"[*] Stage 0 — auditando: {raw_dir}")
    n_ok, n_fail = 0, 0

    for case_dir in scan_raw_cases(raw_dir):
        result = audit_case(case_dir, case_dir.name, modalities, label_file)
        if result.status == "validated":
            print(f"  [+] {result.case_id}: OK (shape {result.shape})")
            n_ok += 1
        else:
            print(f"  [-] {result.case_id}: {result.reason} — {result.details}")
            n_fail += 1

    print(f"\n[*] Resumen: {n_ok} OK, {n_fail} con problemas")


def main():
    parser = argparse.ArgumentParser(description="Auditar datos raw BraTS (Stage 0)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    audit_raw(load_config(args.config))


if __name__ == "__main__":
    main()
