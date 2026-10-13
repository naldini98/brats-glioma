#!/usr/bin/env python3
"""Evalúa val/test hold-out con nnU-Net entrenado (Dice WT/TC/ET).

Uso:
  python scripts/eval_nnunet.py --yes
  python main.py --stage eval --yes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline_io import load_config
from src.eval.run_eval import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval: nnU-Net inferencia + Dice WT/TC/ET")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--yes", action="store_true", help="Sin prompt interactivo")
    parser.add_argument(
        "--subset",
        choices=["val", "test", "both"],
        default="both",
        help="Subconjunto a evaluar (default: both)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if "eval" not in config:
        config["eval"] = {}
    if args.subset == "val":
        config["eval"]["subsets"] = ["val"]
    elif args.subset == "test":
        config["eval"]["subsets"] = ["test"]
    else:
        config["eval"]["subsets"] = ["val", "test"]
    run_eval(config, yes=args.yes)


if __name__ == "__main__":
    main()
