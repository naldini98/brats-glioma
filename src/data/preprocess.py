"""Preprocesamiento — alias de Stage 1.

Stage 1: RAS + resample 1mm + clip percentil + z-score por modalidad/caso.
"""

from src.data.stage_1 import run_stage_1


def run_preprocess(config: dict, *, yes: bool = False, base=None):
    """Compatibilidad con invocaciones antiguas de preprocess."""
    return run_stage_1(config, yes=yes)
