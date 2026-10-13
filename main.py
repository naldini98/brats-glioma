#!/usr/bin/env python3
"""Punto de entrada del pipeline BraTS Glioma Segmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        print(f"[-] Error: No se encontró la configuración en {config_path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="BraTS Glioma Segmentation Pipeline")
    parser.add_argument(
        "--stage",
        choices=[
            "stage_0",
            "stage_1",
            "stage_2",
            "stage_3",
            "stage_4",
            "preprocess",
            "train",
            "eval",
            "all",
        ],
        required=True,
        help="Etapa a ejecutar (preprocess=stage_1, train=stage_4)",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirma la ejecución sin prompt interactivo",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    run_stage_1 = args.stage in ["stage_1", "preprocess", "all"]
    run_stage_4 = args.stage in ["stage_4", "train", "all"]

    if args.stage in ["stage_0", "all"]:
        print("[*] Stage 0: Generación de Volúmenes y Máscaras Base")
        from src.data.stage_0 import run_stage_0

        run_stage_0(config, yes=args.yes)

    if run_stage_1:
        print("[*] Stage 1: Preprocesamiento (RAS + resample + normalización)")
        from src.data.stage_1 import run_stage_1 as _run_stage_1

        _run_stage_1(config, yes=args.yes)

    if args.stage in ["stage_2", "all"]:
        print("[*] Stage 2: Partición reproducible 70/15/15")
        from src.data.stage_2 import run_stage_2

        run_stage_2(config, yes=args.yes)

    if args.stage in ["stage_3", "all"]:
        print("[*] Stage 3: Conversión a formato nnU-Net v2")
        from src.data.stage_3 import run_stage_3

        run_stage_3(config, yes=args.yes)

    if run_stage_4:
        print("[*] Stage 4: Plan/preprocess + Train (nnU-Net v2)")
        from src.models.stage_4 import run_stage_4

        run_stage_4(config, yes=args.yes)

    if args.stage in ["eval", "all"]:
        print("[*] Eval: inferencia nnU-Net + Dice WT/TC/ET (val/test hold-out)")
        from src.eval.run_eval import run_eval

        run_eval(config, yes=args.yes)


if __name__ == "__main__":
    main()
