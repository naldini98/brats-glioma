"""Stage 1: Preprocesamiento espacial e intensidad (RAS + 1mm + z-score).

Entrada: output/stage_0 (solo SUCCESS_GEOMETRIC_APPROVED).
Salida: output/stage_1 con volúmenes normalizados y seg resampleada (nearest).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.data.dicom_geometry import ensure_dir, geometries_match, geometry_tuple, read_nifti, write_nifti
from src.data.intensity_norm import normalize_modality_volumes
from src.data.labels import invalid_seg_values, validate_seg
from src.data.spatial_preprocess import (
    build_reference_grid,
    reorient_to_ras,
    resample_to_reference,
)
from src.data.stage_0 import (
    STATUS_FAIL as STAGE0_FAIL,
    STATUS_OK as STAGE0_OK,
    output_paths_for_case,
    project_root_from_config,
    stage0_paths,
)

STATUS_OK = "SUCCESS_PREPROCESSED"
STATUS_FAIL = "FAILED"


def stage1_paths(config: dict) -> dict[str, Path]:
    root = project_root_from_config(config)
    cfg = config["stage_1"]
    s0 = stage0_paths(config)

    def _p(key: str) -> Path:
        value = Path(cfg[key])
        return value if value.is_absolute() else (root / value)

    return {
        "root": root,
        "input_log": s0["output_log"],
        "input_volumes_dir": s0["volumes_dir"],
        "input_masks_dir": s0["masks_dir"],
        "output_log": _p("output_log"),
        "volumes_dir": _p("volumes_dir"),
        "masks_dir": _p("masks_dir"),
        "audit_report": _p("audit_report"),
    }


def load_approved_case_ids(stage0_log: Path) -> list[str]:
    """Casos aprobados por Stage 0."""
    if not stage0_log.exists():
        raise FileNotFoundError(f"No se encontró log Stage 0: {stage0_log}")
    df = pd.read_csv(stage0_log)
    if "case_id" not in df.columns or "status" not in df.columns:
        raise ValueError("processing_log Stage 0 sin columnas case_id/status")
    approved = df[df["status"] == STAGE0_OK]["case_id"].astype(str).tolist()
    return sorted(approved)


def verify_stage_1_output(
    case_id: str,
    volumes_dir: Path,
    masks_dir: Path,
    modalities: list[str] | None = None,
    label_file: str = "seg",
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    spacing_tol: float = 1e-3,
) -> tuple[str, str, dict]:
    """Gate Stage 1: archivos, geometría melliza, spacing ~1mm, seg uint8 {0,1,2,3}."""
    modalities = modalities or ["t1", "t1ce", "t2", "flair"]
    paths = output_paths_for_case(case_id, volumes_dir, masks_dir, modalities, label_file)

    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return STATUS_FAIL, "missing_nifti", {"missing": missing}

    try:
        images = {name: read_nifti(path) for name, path in paths.items()}
    except Exception as exc:  # noqa: BLE001
        return STATUS_FAIL, "read_error", {"error": str(exc)}

    ordered = [images[m] for m in modalities] + [images[label_file]]
    ok, msg = geometries_match(ordered)
    if not ok:
        return STATUS_FAIL, "geometry_mismatch", {"details": msg}

    ref = images[modalities[0]]
    spacing = tuple(float(s) for s in ref.GetSpacing())
    if any(abs(s - t) > spacing_tol for s, t in zip(spacing, target_spacing)):
        return STATUS_FAIL, "spacing_mismatch", {"spacing": spacing, "expected": target_spacing}

    seg = images[label_file]
    if seg.GetPixelID() != sitk.sitkUInt8:
        return STATUS_FAIL, "seg_dtype", {"pixel_id": int(seg.GetPixelID())}

    seg_arr = np.asarray(sitk.GetArrayFromImage(seg))
    if not validate_seg(seg_arr):
        invalid = sorted(invalid_seg_values(seg_arr))
        return STATUS_FAIL, "corrupt_seg", {"invalid_values": invalid}

    for mod in modalities:
        if images[mod].GetPixelID() not in (sitk.sitkFloat32, sitk.sitkFloat64):
            return STATUS_FAIL, "modality_dtype", {"modality": mod, "pixel_id": int(images[mod].GetPixelID())}

    shapes = {name: geometry_tuple(img)["size"] for name, img in images.items()}
    return (
        STATUS_OK,
        "",
        {
            "shapes": shapes,
            "spacing": spacing,
            "geometry": geometry_tuple(ref),
            "paths": {k: str(v) for k, v in paths.items()},
        },
    )


def process_one_case(
    case_id: str,
    *,
    input_volumes_dir: Path,
    input_masks_dir: Path,
    output_volumes_dir: Path,
    output_masks_dir: Path,
    modalities: list[str],
    label_file: str,
    target_spacing: tuple[float, float, float],
    reference_modality: str,
    mask_source: str,
    mask_threshold: float,
    p_low: float,
    p_high: float,
    eps: float,
) -> dict:
    record: dict = {
        "case_id": case_id,
        "status": STATUS_FAIL,
        "reason": "",
        "details": "",
        "shapes": "",
        "spacing": "",
        "path_t1": "",
        "path_t1ce": "",
        "path_t2": "",
        "path_flair": "",
        "path_seg": "",
    }

    try:
        in_paths = output_paths_for_case(
            case_id, input_volumes_dir, input_masks_dir, modalities, label_file
        )
        missing = [k for k, p in in_paths.items() if not p.exists()]
        if missing:
            record["reason"] = "missing_stage0_input"
            record["details"] = f"Faltan: {missing}"
            return record

        # 1) Cargar
        loaded = {name: read_nifti(path) for name, path in in_paths.items()}

        # 2) Reorientar a RAS (MRI linear, seg nearest)
        reoriented_mri = {
            mod: reorient_to_ras(loaded[mod]) for mod in modalities
        }
        reoriented_seg = reorient_to_ras(loaded[label_file])

        # 3) Grilla de referencia isotrópica desde modalidad de referencia
        if reference_modality not in reoriented_mri:
            record["reason"] = "bad_reference_modality"
            record["details"] = reference_modality
            return record

        ref_grid = build_reference_grid(reoriented_mri[reference_modality], target_spacing)

        resampled_mri = {
            mod: resample_to_reference(
                reoriented_mri[mod], ref_grid, interpolator=sitk.sitkLinear
            )
            for mod in modalities
        }
        resampled_seg = resample_to_reference(
            reoriented_seg,
            ref_grid,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0.0,
        )
        resampled_seg = sitk.Cast(resampled_seg, sitk.sitkUInt8)

        # Validar seg antes de normalizar (nearest puede introducir valores si input corrupto)
        seg_arr = np.asarray(sitk.GetArrayFromImage(resampled_seg))
        if not validate_seg(seg_arr):
            invalid = sorted(invalid_seg_values(seg_arr))
            record["reason"] = "corrupt_seg"
            record["details"] = f"Tras resample: {invalid}"
            return record

        # 4) Normalización: clip percentil + z-score por modalidad/caso
        normalized_mri, _ = normalize_modality_volumes(
            resampled_mri,
            mask_source=mask_source,
            mask_threshold=mask_threshold,
            p_low=p_low,
            p_high=p_high,
            eps=eps,
        )

        # 5) Guardar
        out_paths = output_paths_for_case(
            case_id, output_volumes_dir, output_masks_dir, modalities, label_file
        )
        for mod in modalities:
            write_nifti(normalized_mri[mod], out_paths[mod])
        write_nifti(resampled_seg, out_paths[label_file])

        # 6) Gate
        status, reason, details = verify_stage_1_output(
            case_id,
            output_volumes_dir,
            output_masks_dir,
            modalities,
            label_file,
            target_spacing=target_spacing,
        )
        record["status"] = status
        record["reason"] = reason
        if status == STATUS_OK:
            record["details"] = "RAS + 1mm + clip/z-score OK"
            shapes = details.get("shapes") or {}
            ref_shape = shapes.get(modalities[0])
            if ref_shape:
                record["shapes"] = f"{ref_shape[2]}x{ref_shape[1]}x{ref_shape[0]}"
            record["spacing"] = "x".join(str(round(s, 4)) for s in details.get("spacing", ()))
        else:
            record["details"] = str(details)

        for mod in modalities:
            record[f"path_{mod}"] = str(out_paths[mod])
        record[f"path_{label_file}"] = str(out_paths[label_file])
        return record

    except Exception as exc:  # noqa: BLE001
        record["reason"] = "exception"
        record["details"] = str(exc)
        return record


def run_stage_1(config: dict, *, yes: bool = False) -> pd.DataFrame:
    """Orquesta Stage 1 completo."""
    paths = stage1_paths(config)
    cfg = config["stage_1"]
    modalities: list[str] = list(cfg["modalities"])
    label_file: str = cfg.get("label_file", "seg")
    target_spacing = tuple(float(x) for x in cfg["target_spacing"])
    max_cases = cfg.get("max_cases")

    norm_cfg = cfg.get("normalization", {})
    p_low = float(norm_cfg.get("percentile_low", 0.5))
    p_high = float(norm_cfg.get("percentile_high", 99.5))
    eps = float(norm_cfg.get("eps", 1e-8))
    mask_source = str(norm_cfg.get("mask_source", "flair"))
    mask_threshold = float(norm_cfg.get("mask_threshold", 0.0))
    reference_modality = str(cfg.get("reference_modality", "t1"))

    ensure_dir(paths["volumes_dir"])
    ensure_dir(paths["masks_dir"])
    ensure_dir(paths["output_log"].parent)

    case_ids = load_approved_case_ids(paths["input_log"])
    if max_cases is not None:
        case_ids = case_ids[: int(max_cases)]

    print("[*] Stage 1: Preprocesamiento (RAS + resample + normalización MRI)")
    print(f"    input volumes: {paths['input_volumes_dir']}")
    print(f"    input masks:   {paths['input_masks_dir']}")
    print(f"    output:        {paths['volumes_dir']}")
    print(f"    casos:         {len(case_ids)} (aprobados Stage 0)")
    print(f"    orientación:   RAS")
    print(f"    spacing:       {target_spacing} mm")
    print(f"    normalización: clip [{p_low}, {p_high}] + z-score por modalidad/caso")
    print(f"    máscara stats: {mask_source} (threshold>{mask_threshold})")
    print(f"    seg interp:    nearest | MRI interp: linear")

    if not yes:
        answer = input("Continuar? [y/N]: ").strip().lower()
        if answer not in {"y", "yes", "s", "si"}:
            print("[-] Cancelado por el usuario.")
            return pd.DataFrame()

    records: list[dict] = []
    for i, case_id in enumerate(case_ids, start=1):
        print(f"[{i}/{len(case_ids)}] {case_id} ...", flush=True)
        rec = process_one_case(
            case_id,
            input_volumes_dir=paths["input_volumes_dir"],
            input_masks_dir=paths["input_masks_dir"],
            output_volumes_dir=paths["volumes_dir"],
            output_masks_dir=paths["masks_dir"],
            modalities=modalities,
            label_file=label_file,
            target_spacing=target_spacing,
            reference_modality=reference_modality,
            mask_source=mask_source,
            mask_threshold=mask_threshold,
            p_low=p_low,
            p_high=p_high,
            eps=eps,
        )
        records.append(rec)
        flag = "+" if rec["status"] == STATUS_OK else "-"
        print(f"  [{flag}] {rec['case_id']}: {rec['status']} {rec['reason']}".rstrip())

    log_df = pd.DataFrame(records)
    log_df.to_csv(paths["output_log"], index=False)

    n_ok = int((log_df["status"] == STATUS_OK).sum()) if not log_df.empty else 0
    n_fail = len(log_df) - n_ok
    print(f"[*] Log: {paths['output_log']}")
    print(f"[*] Resumen: {n_ok} {STATUS_OK}, {n_fail} {STATUS_FAIL}")
    return log_df
