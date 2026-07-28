"""Entorno y helpers CLI para nnU-Net v2 (Stage 3)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


def resolve_under_root(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def apply_nnunet_env(
    *,
    nnunet_raw: Path,
    nnunet_preprocessed: Path,
    nnunet_results: Path,
) -> dict[str, str]:
    """Define nnUNet_raw / nnUNet_preprocessed / nnUNet_results en el proceso."""
    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw.resolve())
    env["nnUNet_preprocessed"] = str(nnunet_preprocessed.resolve())
    env["nnUNet_results"] = str(nnunet_results.resolve())
    return env


def detect_device(preferred: str = "auto") -> str:
    """Resuelve device: auto → cuda si hay GPU, si no cpu."""
    pref = (preferred or "auto").strip().lower()
    if pref in {"cuda", "cpu", "mps"}:
        if pref == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "device=cuda en config pero torch.cuda.is_available()=False"
                    )
            except ImportError as exc:
                raise RuntimeError("PyTorch no disponible para device=cuda") from exc
        return pref

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def find_nnunet_command(name: str) -> str:
    """Localiza ejecutable nnU-Net (PATH o Scripts del intérprete actual)."""
    found = shutil.which(name)
    if found:
        return found
    # Fallback: Scripts junto a python.exe (conda en Windows)
    scripts = Path(__import__("sys").executable).resolve().parent / "Scripts" / f"{name}.exe"
    if scripts.exists():
        return str(scripts)
    scripts2 = Path(__import__("sys").executable).resolve().parent / name
    if scripts2.exists():
        return str(scripts2)
    raise FileNotFoundError(
        f"No se encontró '{name}'. Activá el entorno conda brats-glioma e instalá nnunetv2."
    )


def run_command(
    cmd: Sequence[str],
    *,
    env: dict[str, str],
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Ejecuta comando nnU-Net heredando stdout/stderr al terminal."""
    print(f"[*] CMD: {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(
        list(cmd),
        env=env,
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
    )


def parse_folds(folds_cfg) -> list[int]:
    """Interpreta folds: [0], 'all', 0, etc."""
    if folds_cfg is None:
        return [0]
    if isinstance(folds_cfg, str):
        if folds_cfg.strip().lower() == "all":
            return [0, 1, 2, 3, 4]
        return [int(folds_cfg)]
    if isinstance(folds_cfg, int):
        return [int(folds_cfg)]
    return [int(x) for x in folds_cfg]


def expected_results_dir(
    nnunet_results: Path,
    *,
    dataset_name: str,
    trainer: str,
    plans: str,
    configuration: str,
) -> Path:
    """Carpeta típica de resultados nnU-Net v2."""
    return (
        nnunet_results
        / f"{dataset_name}"
        / f"{trainer}__{plans}__{configuration}"
    )


def fold_checkpoint_path(results_dir: Path, fold: int, name: str = "checkpoint_final.pth") -> Path:
    return results_dir / f"fold_{fold}" / name


def plans_json_path(nnunet_preprocessed: Path, dataset_name: str, plans: str = "nnUNetPlans") -> Path:
    return nnunet_preprocessed / dataset_name / f"{plans}.json"
