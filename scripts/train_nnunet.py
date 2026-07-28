#!/usr/bin/env python3
"""Entrena nnU-Net v2 (wrapper de Stage 3).

Uso:
  python scripts/train_nnunet.py --yes
  python main.py --stage stage_3 --yes
  python main.py --stage train --yes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline_io import load_config
from src.models.stage_3 import run_stage_3


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: nnU-Net plan + train")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--yes", action="store_true", help="Sin prompt interactivo")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Solo plan_and_preprocess (sin train)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Solo train (asume plan ya hecho)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if "stage_3" not in config:
        config["stage_3"] = {}
    if args.plan_only:
        config["stage_3"]["run_plan_and_preprocess"] = True
        config["stage_3"]["run_train"] = False
    if args.train_only:
        config["stage_3"]["run_plan_and_preprocess"] = False
        config["stage_3"]["run_train"] = True
    run_stage_3(config, yes=args.yes)


if __name__ == "__main__":
    main()
