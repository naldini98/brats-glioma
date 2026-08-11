#!/usr/bin/env python3
"""Convierte Stage 1 + split al formato nnU-Net v2 (wrapper de Stage 3).

Uso:
  python scripts/convert_to_nnunet.py --yes
  python main.py --stage stage_3 --yes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline_io import load_config
from src.data.stage_3 import run_stage_3


def main() -> None:
    parser = argparse.ArgumentParser(description="Convertir → nnU-Net v2 (Stage 3)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirma la ejecución sin prompt interactivo",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_stage_3(config, yes=args.yes)


if __name__ == "__main__":
    main()
