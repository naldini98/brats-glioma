"""Compatibilidad: Stage 4 reemplaza el antiguo Stage 3 (plan/train nnU-Net)."""

from src.models.stage_4 import (
    build_plan_command,
    build_train_command,
    run_stage_3,
    run_stage_4,
    stage3_paths,
    stage4_paths,
    verify_fold_checkpoint,
    verify_plans,
    verify_stage2_dataset,
    verify_stage3_dataset,
)

__all__ = [
    "build_plan_command",
    "build_train_command",
    "run_stage_3",
    "run_stage_4",
    "stage3_paths",
    "stage4_paths",
    "verify_fold_checkpoint",
    "verify_plans",
    "verify_stage2_dataset",
    "verify_stage3_dataset",
]
