"""Validaciones estructurales del Stage 0 (Ingestion & Structural Audit).

No modifica datos; solo audita geometría y etiquetas BraTS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np

from src.data.labels import invalid_seg_values, validate_seg

RejectionReason = Literal[
    "missing_files",
    "geometry_shape",
    "geometry_affine",
    "corrupt_seg",
]

AuditStatus = Literal["validated", "rejected"]


@dataclass
class CaseAuditResult:
    case_id: str
    status: AuditStatus
    reason: RejectionReason | None = None
    missing_files: list[str] = field(default_factory=list)
    shape: tuple[int, ...] | None = None
    invalid_seg_values: list[int] = field(default_factory=list)
    details: str = ""
    raw_dir: str = ""
    cases_dir: str = ""

    def to_log_record(self) -> dict:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "reason": self.reason or "",
            "missing_files": ",".join(self.missing_files),
            "shape": "x".join(map(str, self.shape)) if self.shape else "",
            "invalid_seg_values": ",".join(map(str, self.invalid_seg_values)),
            "details": self.details,
            "raw_dir": self.raw_dir,
            "cases_dir": self.cases_dir,
        }


def expected_case_files(case_id: str, modalities: list[str], label_file: str) -> dict[str, Path]:
    """Devuelve los 5 archivos obligatorios por modalidad/etiqueta (sin resolver directorio)."""
    files = {mod: Path(f"{case_id}-{mod}.nii.gz") for mod in modalities}
    files[label_file] = Path(f"{case_id}-{label_file}.nii.gz")
    return files


def check_required_files(
    case_dir: Path,
    case_id: str,
    modalities: list[str],
    label_file: str,
) -> tuple[bool, list[str], dict[str, Path]]:
    """Verifica presencia de las 4 modalidades + máscara _seg."""
    missing: list[str] = []
    resolved: dict[str, Path] = {}

    for mod in modalities:
        path = case_dir / f"{case_id}-{mod}.nii.gz"
        resolved[mod] = path
        if not path.exists():
            missing.append(mod)

    seg_path = case_dir / f"{case_id}-{label_file}.nii.gz"
    resolved[label_file] = seg_path
    if not seg_path.exists():
        missing.append(label_file)

    return len(missing) == 0, missing, resolved


def validate_modality_geometry(modality_paths: dict[str, Path]) -> tuple[bool, str, dict]:
    """Comprueba dimensiones y matriz affine idénticas entre modalidades MRI.

    Returns:
        (ok, reason_code, details) donde reason_code es 'ok', 'geometry_shape' o 'geometry_affine'.
    """
    if not modality_paths:
        return False, "geometry_shape", {"error": "no modality paths provided"}

    shapes: dict[str, tuple[int, ...]] = {}
    affines: dict[str, np.ndarray] = {}
    reference_mod = next(iter(modality_paths))

    for mod, path in modality_paths.items():
        img = nib.load(str(path))
        shapes[mod] = tuple(int(s) for s in img.shape[:3])
        affines[mod] = np.asarray(img.affine)

    ref_shape = shapes[reference_mod]
    ref_affine = affines[reference_mod]

    shape_mismatches = {mod: shapes[mod] for mod in modality_paths if shapes[mod] != ref_shape}
    if shape_mismatches:
        return (
            False,
            "geometry_shape",
            {
                "reference_modality": reference_mod,
                "reference_shape": ref_shape,
                "mismatches": shape_mismatches,
            },
        )

    for mod, affine in affines.items():
        if not np.array_equal(affine, ref_affine):
            return (
                False,
                "geometry_affine",
                {
                    "reference_modality": reference_mod,
                    "mismatched_modality": mod,
                    "reference_shape": ref_shape,
                },
            )

    return True, "ok", {"reference_modality": reference_mod, "shape": ref_shape}


def validate_seg_volume(seg_path: Path) -> tuple[bool, list[int]]:
    """Filtro biológico cero: solo clases BraTS {0, 1, 2, 4}. La clase 3 no existe."""
    seg_data = np.asanyarray(nib.load(str(seg_path)).dataobj)
    invalid = sorted(invalid_seg_values(seg_data))
    return validate_seg(seg_data), invalid


def audit_case(
    case_dir: Path,
    case_id: str,
    modalities: list[str],
    label_file: str,
    cases_root: Path | None = None,
) -> CaseAuditResult:
    """Auditoría estructural completa de un caso BraTS (Stage 0)."""
    raw_dir = str(case_dir.resolve())
    cases_dir = str((cases_root / case_id).resolve()) if cases_root else ""

    files_ok, missing, resolved = check_required_files(case_dir, case_id, modalities, label_file)
    if not files_ok:
        return CaseAuditResult(
            case_id=case_id,
            status="rejected",
            reason="missing_files",
            missing_files=missing,
            details=f"Faltan {len(missing)} archivo(s) obligatorio(s)",
            raw_dir=raw_dir,
        )

    modality_paths = {mod: resolved[mod] for mod in modalities}
    geom_ok, geom_reason, geom_details = validate_modality_geometry(modality_paths)
    if not geom_ok:
        shape = geom_details.get("reference_shape") or geom_details.get("shape")
        return CaseAuditResult(
            case_id=case_id,
            status="rejected",
            reason=geom_reason,  # type: ignore[arg-type]
            shape=tuple(shape) if shape else None,
            details=_geometry_details_message(geom_reason, geom_details),
            raw_dir=raw_dir,
        )

    seg_ok, invalid_values = validate_seg_volume(resolved[label_file])
    ref_shape = geom_details["shape"]
    if not seg_ok:
        return CaseAuditResult(
            case_id=case_id,
            status="rejected",
            reason="corrupt_seg",
            shape=ref_shape,
            invalid_seg_values=invalid_values,
            details=f"Clases inválidas en _seg: {invalid_values}",
            raw_dir=raw_dir,
        )

    return CaseAuditResult(
        case_id=case_id,
        status="validated",
        shape=ref_shape,
        raw_dir=raw_dir,
        cases_dir=cases_dir,
        details="Auditoría estructural OK",
    )


def _geometry_details_message(reason: str, details: dict) -> str:
    if reason == "geometry_shape":
        ref = details["reference_modality"]
        ref_shape = details["reference_shape"]
        mismatches = ", ".join(f"{mod}={shape}" for mod, shape in details["mismatches"].items())
        return f"Shape distinto a {ref} {ref_shape}: {mismatches}"
    if reason == "geometry_affine":
        return (
            f"Affine distinto entre {details['reference_modality']} "
            f"y {details['mismatched_modality']}"
        )
    return reason


def scan_raw_cases(raw_dir: Path) -> list[Path]:
    """Lista directorios de paciente en 00_raw."""
    if not raw_dir.exists():
        return []
    return sorted(p for p in raw_dir.iterdir() if p.is_dir())
