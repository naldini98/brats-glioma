"""Utilidades de E/S compartidas por el pipeline."""

from pathlib import Path

import yaml


def load_config(config_path: str | Path = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config: dict, key: str, base: Path | None = None) -> Path:
    """Resuelve una ruta relativa definida en config['paths']."""
    root = base or Path.cwd()
    return (root / config["paths"][key]).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
