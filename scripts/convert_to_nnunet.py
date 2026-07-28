#!/usr/bin/env python3
"""Convierte Stage 1 al formato nnU-Net v2 (wrapper de Stage 2).

Uso:
  python scripts/convert_to_nnunet.py --yes
  python main.py --stage stage_2 --yes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline_io import load_config
from src.data.stage_2 import run_stage_2


def main() -> None:
    parser = argparse.ArgumentParser(description="Convertir Stage 1 → nnU-Net v2 (Stage 2)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirma la ejecución sin prompt interactivo",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_stage_2(config, yes=args.yes)


if __name__ == "__main__":
    main()
