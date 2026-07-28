"""Stage 0: Generación de Volúmenes y Máscaras Base (Decathlon → NIfTI mellizos).

Materia prima: Task01_BrainTumour (imágenes 4D + máscaras 3D).
Espacio NATIVO: sin RAS, sin resampling, sin normalización.
Clases nativas Decathlon: {0, 1, 2, 3}.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import SimpleITK as sitk

from src.data.dicom_geometry import ensure_dir, geometries_match, geometry_tuple, read_nifti
from src.data.mask_generator import process_and_save_mask
from src.data.preprocessor import list_training_cases, load_and_split_case, save_modality_volumes

STATUS_OK = "SUCCESS_GEOMETRIC_APPROVED"
STATUS_FAIL = "FAILED"


def project_root_from_config(config: dict) -> Path:
    root = config.get("project_root")
    if root:
        return Path(root)
    return Path(__file__).resolve().parents[3]


def stage0_paths(config: dict) -> dict[str, Path]:
    root = project_root_from_config(config)
    cfg = config["stage_0"]

    def _p(key: str) -> Path:
        value = Path(cfg[key])
        return value if value.is_absolute() else (root / value)

    decathlon = _p("decathlon_root")
    return {
        "root": root,
        "decathlon_root": decathlon,
        "images_dir": decathlon / cfg.get("images_subdir", "imagesTr"),
        "labels_dir": decathlon / cfg.get("labels_subdir", "labelsTr"),
        "dataset_json": decathlon / cfg.get("dataset_json", "dataset.json"),
        "output_log": _p("output_log"),
        "volumes_dir": _p("volumes_dir"),
        "masks_dir": _p("masks_dir"),
        "audit_report": _p("audit_report"),
    }


def output_paths_for_case(
    case_id: str,
    volumes_dir: Path,
    masks_dir: Path,
    modalities: list[str],
    label_file: str = "seg",
) -> dict[str, Path]:
    folder = f"volume_{case_id}"
    paths = {
        mod: volumes_dir / folder / f"{folder}-{mod}.nii.gz" for mod in modalities
    }
    paths[label_file] = masks_dir / folder / f"{folder}-{label_file}.nii.gz"
    return paths


def verify_stage_0_output(
    case_id: str,
    volumes_dir: Path,
    masks_dir: Path,
    modalities: list[str] | None = None,
    label_file: str = "seg",
) -> tuple[str, str, dict]:
    """Gate: 5 NIfTI existen, mismo size y misma geometría SITK."""
    modalities = modalities or ["t1", "t1ce", "t2", "flair"]
    paths = output_paths_for_case(case_id, volumes_dir, masks_dir, modalities, label_file)

    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return (
            STATUS_FAIL,
            "missing_nifti",
            {"missing": missing, "paths": {k: str(v) for k, v in paths.items()}},
        )

    try:
        images = {name: read_nifti(path) for name, path in paths.items()}
    except Exception as exc:  # noqa: BLE001
        return STATUS_FAIL, "read_error", {"error": str(exc)}

    ordered = [images[m] for m in modalities] + [images[label_file]]
    ok, msg = geometries_match(ordered)
    if not ok:
        return STATUS_FAIL, "geometry_mismatch", {"details": msg}

    seg = images[label_file]
    if seg.GetPixelID() != sitk.sitkUInt8:
        return STATUS_FAIL, "seg_dtype", {"pixel_id": int(seg.GetPixelID())}

    ref_geom = geometry_tuple(images[modalities[0]])
    shapes = {name: geometry_tuple(img)["size"] for name, img in images.items()}
    return (
        STATUS_OK,
        "",
        {
            "shapes": shapes,
            "geometry": ref_geom,
            "paths": {k: str(v) for k, v in paths.items()},
        },
    )


def process_one_case(
    case: dict,
    *,
    volumes_dir: Path,
    masks_dir: Path,
    modalities: list[str],
    label_file: str,
    channel_to_modality: dict[int, str],
) -> dict:
    case_id = case["case_id"]
    record: dict = {
        "case_id": case_id,
        "status": STATUS_FAIL,
        "reason": "",
        "details": "",
        "shapes": "",
        "path_t1": "",
        "path_t1ce": "",
        "path_t2": "",
        "path_flair": "",
        "path_seg": "",
    }

    try:
        if not case["image_path"].exists():
            record["reason"] = "missing_image"
            record["details"] = str(case["image_path"])
            return record
        if not case["label_path"].exists():
            record["reason"] = "missing_label"
            record["details"] = str(case["label_path"])
            return record

        modality_imgs, reference_3d = load_and_split_case(
            case["image_path"], channel_to_modality=channel_to_modality
        )
        for mod in modalities:
            if mod not in modality_imgs:
                record["reason"] = "missing_modality"
                record["details"] = f"Modalidad ausente tras split: {mod}"
                return record

        vol_paths = save_modality_volumes(
            modality_imgs, case_id, volumes_dir, modality_order=modalities
        )
        seg_path, _ = process_and_save_mask(
            case["label_path"],
            case_id,
            masks_dir,
            reference_3d,
            label_file=label_file,
        )

        status, reason, details = verify_stage_0_output(
            case_id, volumes_dir, masks_dir, modalities, label_file
        )
        record["status"] = status
        record["reason"] = reason
        if status == STATUS_OK:
            record["details"] = "Geometría melliza OK (espacio nativo Decathlon)"
            shapes = details.get("shapes") or {}
            ref = shapes.get(modalities[0])
            if ref:
                # SITK size = (X, Y, Z) → reportar ZxYxX
                record["shapes"] = f"{ref[2]}x{ref[1]}x{ref[0]}"
        else:
            record["details"] = str(details)

        for mod in modalities:
            record[f"path_{mod}"] = str(vol_paths[mod])
        record[f"path_{label_file}"] = str(seg_path)
        return record

    except Exception as exc:  # noqa: BLE001
        record["reason"] = "exception"
        record["details"] = str(exc)
        return record


def run_stage_0(config: dict, *, yes: bool = False) -> pd.DataFrame:
    """Orquesta Stage 0 completo y escribe processing_log.csv."""
    paths = stage0_paths(config)
    cfg = config["stage_0"]
    modalities: list[str] = list(cfg["modalities"])
    label_file: str = cfg.get("label_file", "seg")
    channel_to_modality = {int(k): str(v) for k, v in cfg["channel_to_modality"].items()}
    max_cases = cfg.get("max_cases")

    ensure_dir(paths["volumes_dir"])
    ensure_dir(paths["masks_dir"])
    ensure_dir(paths["output_log"].parent)

    if not paths["decathlon_root"].exists():
        raise FileNotFoundError(f"No se encontró Decathlon: {paths['decathlon_root']}")

    cases = list_training_cases(paths["decathlon_root"], paths["dataset_json"])
    if max_cases is not None:
        cases = cases[: int(max_cases)]

    print("[*] Stage 0: Generación de Volúmenes y Máscaras Base (Decathlon)")
    print(f"    input:  {paths['decathlon_root']}")
    print(f"    volumes:{paths['volumes_dir']}")
    print(f"    masks:  {paths['masks_dir']}")
    print(f"    casos:  {len(cases)}")
    print("    geometría: NATIVA del volumen 4D (sin RAS / resample / z-score)")
    print("    clases seg: {0,1,2,3} (Decathlon nativo, sin remap)")
    print(f"    canales: {channel_to_modality}")

    if not yes:
        answer = input("Continuar? [y/N]: ").strip().lower()
        if answer not in {"y", "yes", "s", "si"}:
            print("[-] Cancelado por el usuario.")
            return pd.DataFrame()

    records: list[dict] = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case['case_id']} ...", flush=True)
        rec = process_one_case(
            case,
            volumes_dir=paths["volumes_dir"],
            masks_dir=paths["masks_dir"],
            modalities=modalities,
            label_file=label_file,
            channel_to_modality=channel_to_modality,
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
