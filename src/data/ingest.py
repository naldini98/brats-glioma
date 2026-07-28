"""Stage 0: Ingestion & Structural Audit (Materia Prima → Datos Validados).

Gatekeeper clínico del pipeline. No modifica data/00_raw/; audita geometría y etiquetas.
Los casos validados se registran en data/01_cases/ y en ingest_log.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.pipeline_io import ensure_dir, resolve_path
from src.data.validation import CaseAuditResult, audit_case, scan_raw_cases


def discover_cases(raw_dir: Path, modalities: list[str], label_file: str) -> list[dict]:
    """Compatibilidad: devuelve solo casos que pasarían la auditoría estructural."""
    cases = []
    cases_root = raw_dir.parent / "01_cases"

    for case_dir in scan_raw_cases(raw_dir):
        result = audit_case(
            case_dir,
            case_dir.name,
            modalities=modalities,
            label_file=label_file,
            cases_root=cases_root,
        )
        if result.status != "validated":
            continue

        files = {mod: case_dir / f"{case_dir.name}-{mod}.nii.gz" for mod in modalities}
        seg_path = case_dir / f"{case_dir.name}-{label_file}.nii.gz"
        cases.append({"case_id": result.case_id, "modalities": files, "seg": seg_path})

    return cases


def run_ingest(config: dict, base: Path | None = None) -> pd.DataFrame:
    """Ejecuta Stage 0: escanea 00_raw, audita y escribe ingest_log.csv."""
    root = base or Path.cwd()
    raw_dir = resolve_path(config, "raw", root)
    cases_dir = ensure_dir(resolve_path(config, "cases", root))
    log_path = resolve_path(config, "log", root)

    ingest_cfg = config["ingest"]
    modalities: list[str] = ingest_cfg["modalities"]
    label_file: str = ingest_cfg["label_file"]

    results: list[CaseAuditResult] = []
    for case_dir in scan_raw_cases(raw_dir):
        result = audit_case(
            case_dir,
            case_dir.name,
            modalities=modalities,
            label_file=label_file,
            cases_root=cases_dir,
        )
        results.append(result)

        if result.status == "validated":
            ensure_dir(cases_dir / result.case_id)

    records = [r.to_log_record() for r in results]
    df = pd.DataFrame(records)

    if not df.empty:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(log_path, index=False)

    n_valid = sum(1 for r in results if r.status == "validated")
    n_rejected = len(results) - n_valid
    print(f"[+] Stage 0 — casos escaneados: {len(results)}")
    print(f"    validados: {n_valid} | rechazados: {n_rejected}")
    print(f"    log: {log_path}")

    for result in results:
        if result.status == "rejected":
            print(f"    [-] {result.case_id}: {result.reason} — {result.details}")

    return df
