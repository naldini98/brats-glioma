#!/usr/bin/env python3
"""Auditoría de data/02_preprocessed: shape, spacing y clases BraTS."""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.labels import validate_seg
from src.data.pipeline_io import load_config, resolve_path

VALID_CLASSES = {0, 1, 2, 4}


def audit_preprocessed(config: dict, base: Path | None = None) -> None:
    root = base or Path.cwd()
    preprocessed = resolve_path(config, "preprocessed", root)
    target_spacing = config["preprocess"]["spacing"]

    if not preprocessed.exists():
        print(f"[-] Directorio preprocesado no encontrado: {preprocessed}")
        return

    print(f"[*] Auditando: {preprocessed}")
    print(f"[*] Spacing objetivo: {target_spacing}")

    case_dirs = sorted(p for p in preprocessed.iterdir() if p.is_dir())
    if not case_dirs:
        print("[-] No hay casos preprocesados.")
        return

    for case_dir in case_dirs:
        seg_files = list(case_dir.glob("*seg*.nii.gz"))
        if not seg_files:
            print(f"  [-] {case_dir.name}: sin máscara de segmentación")
            continue

        seg_nii = nib.load(str(seg_files[0]))
        seg_data = np.asanyarray(seg_nii.dataobj)
        spacing = seg_nii.header.get_zooms()[:3]

        valid = validate_seg(seg_data)
        status = "OK" if valid else f"clases inválidas {set(np.unique(seg_data)) - VALID_CLASSES}"
        print(f"  [{'+'if valid else '-'}] {case_dir.name}: shape={seg_data.shape}, spacing={spacing}, {status}")


def main():
    parser = argparse.ArgumentParser(description="Auditar datos preprocesados")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    audit_preprocessed(load_config(args.config))


if __name__ == "__main__":
    main()
