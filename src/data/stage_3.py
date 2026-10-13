"""Stage 3: Conversión a formato nnU-Net v2 (Dataset101_BraTS).

Entrada: output/stage_1 (solo subset train del split Stage 2).
Salida: output/nnunet_raw/Dataset101_BraTS/imagesTr + labelsTr.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.data.dicom_geometry import ensure_dir, geometries_match, geometry_tuple, read_nifti
from src.data.labels import invalid_seg_values, validate_seg
from src.data.nnunet_convert import (
    build_dataset_json,
    copy_nifti,
    nnunet_case_id,
    nnunet_paths_for_case,
    prepare_dataset_dirs,
    write_dataset_json,
)
from src.data.stage_0 import output_paths_for_case, project_root_from_config
from src.data.stage_1 import stage1_paths
from src.data.stage_2 import load_split_json, stage2_paths as split_stage2_paths

STATUS_OK = "SUCCESS_CONVERTED"
STATUS_FAIL = "FAILED"


def stage3_paths(config: dict) -> dict[str, Path]:
    root = project_root_from_config(config)
    cfg = config["stage_3"]
    s1 = stage1_paths(config)
    s2 = split_stage2_paths(config)
    nn_cfg = config.get("nnunet", {})

    def _p(key: str, default: str) -> Path:
        value = Path(cfg.get(key, default))
        return value if value.is_absolute() else (root / value)

    nnunet_raw = cfg.get("nnunet_raw") or config.get("paths", {}).get("nnunet_raw", "output/nnunet_raw")
    nnunet_raw_path = Path(nnunet_raw)
    if not nnunet_raw_path.is_absolute():
        nnunet_raw_path = root / nnunet_raw_path

    dataset_id = int(nn_cfg.get("dataset_id", 101))
    dataset_name = str(nn_cfg.get("dataset_name", "Dataset101_BraTS"))
    dirs = prepare_dataset_dirs(nnunet_raw_path, dataset_id, dataset_name)

    return {
        "root": root,
        "split_json": s2["split_json"],
        "input_log": s1["output_log"],
        "input_volumes_dir": s1["volumes_dir"],
        "input_masks_dir": s1["masks_dir"],
        "output_log": _p("output_log", "output/stage_3/conversion_log.csv"),
        "audit_report": _p("audit_report", "output/stage_3/reporte_auditoria_stage_3.md"),
        "nnunet_raw": nnunet_raw_path,
        "dataset_folder": dirs["dataset_folder"],
        "images_tr": dirs["images_tr"],
        "labels_tr": dirs["labels_tr"],
        "images_ts": dirs["images_ts"],
        "dataset_json": dirs["dataset_json"],
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
    }


def load_train_case_ids(split_json: Path) -> list[str]:
    """IDs del subset train definido en Stage 2."""
    data = load_split_json(split_json)
    return sorted(str(c) for c in data["train"])


def verify_stage_3_output(
    case_id: str,
    images_tr: Path,
    labels_tr: Path,
    modalities: list[str] | None = None,
) -> tuple[str, str, dict]:
    """Gate Stage 3: N canales + label, geometría melliza, seg {0,1,2,3}."""
    modalities = modalities or ["t1", "t1ce", "t2", "flair"]
    n_ch = len(modalities)
    paths = nnunet_paths_for_case(case_id, images_tr, labels_tr, n_ch)

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

    ordered = [images[f"channel_{i}"] for i in range(n_ch)] + [images["label"]]
    ok, msg = geometries_match(ordered)
    if not ok:
        return STATUS_FAIL, "geometry_mismatch", {"details": msg}

    seg = images["label"]
    if seg.GetPixelID() != sitk.sitkUInt8:
        return STATUS_FAIL, "seg_dtype", {"pixel_id": int(seg.GetPixelID())}

    seg_arr = np.asarray(sitk.GetArrayFromImage(seg))
    if not validate_seg(seg_arr):
        invalid = sorted(invalid_seg_values(seg_arr))
        return STATUS_FAIL, "corrupt_seg", {"invalid_values": invalid}

    for i in range(n_ch):
        img = images[f"channel_{i}"]
        if img.GetPixelID() not in (sitk.sitkFloat32, sitk.sitkFloat64):
            return (
                STATUS_FAIL,
                "modality_dtype",
                {"channel": i, "pixel_id": int(img.GetPixelID())},
            )

    ref = images["channel_0"]
    return (
        STATUS_OK,
        "",
        {
            "case_id": nnunet_case_id(case_id),
            "n_channels": n_ch,
            "geometry": geometry_tuple(ref),
            "paths": {k: str(v) for k, v in paths.items()},
        },
    )


def process_one_case(
    case_id: str,
    *,
    input_volumes_dir: Path,
    input_masks_dir: Path,
    images_tr: Path,
    labels_tr: Path,
    modalities: list[str],
    label_file: str,
) -> dict:
    cid = nnunet_case_id(case_id)
    record: dict = {
        "case_id": cid,
        "status": STATUS_FAIL,
        "reason": "",
        "details": "",
        "n_channels": len(modalities),
        "path_label": "",
    }
    for i, mod in enumerate(modalities):
        record[f"path_{mod}"] = ""
        record[f"channel_{i}"] = mod

    try:
        in_paths = output_paths_for_case(
            case_id, input_volumes_dir, input_masks_dir, modalities, label_file
        )
        missing = [k for k, p in in_paths.items() if not p.exists()]
        if missing:
            record["reason"] = "missing_stage1_input"
            record["details"] = f"Faltan: {missing}"
            return record

        out_paths = nnunet_paths_for_case(cid, images_tr, labels_tr, len(modalities))

        for i, mod in enumerate(modalities):
            dst = copy_nifti(in_paths[mod], out_paths[f"channel_{i}"])
            record[f"path_{mod}"] = str(dst)

        dst_label = copy_nifti(in_paths[label_file], out_paths["label"])
        record["path_label"] = str(dst_label)

        status, reason, details = verify_stage_3_output(
            cid, images_tr, labels_tr, modalities
        )
        record["status"] = status
        record["reason"] = reason
        if status == STATUS_OK:
            record["details"] = "nnU-Net packaging OK"
            geom = details.get("geometry") or {}
            size = geom.get("size")
            if size:
                record["details"] = f"nnU-Net OK size={size}"
        else:
            record["details"] = str(details)
        return record

    except Exception as exc:  # noqa: BLE001
        record["reason"] = "exception"
        record["details"] = str(exc)
        return record


def write_stage3_dataset_json(
    config: dict,
    paths: dict[str, Path],
    num_training: int,
    modalities: list[str],
) -> Path:
    labels_cfg = config.get("labels", {}).get("classes", {
        "background": 0,
        "edema": 1,
        "non_enhancing": 2,
        "enhancing": 3,
    })
    project = config.get("project", {})
    meta = build_dataset_json(
        modalities=modalities,
        labels=labels_cfg,
        num_training=num_training,
        num_test=None,
        dataset_name=str(paths["dataset_name"]),
        release=str(project.get("edition", "1.0")),
    )
    return write_dataset_json(paths["dataset_json"], meta)


def run_stage_3(config: dict, *, yes: bool = False) -> pd.DataFrame:
    """Orquesta Stage 3: conversión nnU-Net solo del subset train."""
    paths = stage3_paths(config)
    cfg = config["stage_3"]
    modalities: list[str] = list(cfg.get("modalities") or config["stage_1"]["modalities"])
    label_file: str = cfg.get("label_file", "seg")
    max_cases = cfg.get("max_cases")

    if not paths["split_json"].exists():
        raise FileNotFoundError(
            f"Falta split.json de Stage 2: {paths['split_json']}. Corré stage_2 primero."
        )

    case_ids = load_train_case_ids(paths["split_json"])
    if max_cases is not None:
        case_ids = case_ids[: int(max_cases)]

    ensure_dir(paths["images_tr"])
    ensure_dir(paths["labels_tr"])
    ensure_dir(paths["output_log"].parent)

    for old in paths["images_tr"].glob("*"):
        if old.is_file():
            old.unlink()
    for old in paths["labels_tr"].glob("*"):
        if old.is_file():
            old.unlink()

    split_data = load_split_json(paths["split_json"])
    print("[*] Stage 3: Conversión a formato nnU-Net v2 (solo train split)")
    print(f"    input volumes: {paths['input_volumes_dir']}")
    print(f"    input masks:   {paths['input_masks_dir']}")
    print(f"    split.json:    {paths['split_json']}")
    print(f"    dataset:       {paths['dataset_folder']}")
    print(f"    casos train:   {len(case_ids)} / {split_data.get('total', '?')} total")
    print(f"    val (hold):    {len(split_data.get('val', []))}  test (hold): {len(split_data.get('test', []))}")
    print(f"    canales:       {list(enumerate(modalities))} → _0000.._{len(modalities)-1:04d}")
    print(f"    dataset.json:  {paths['dataset_json']}")

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
            images_tr=paths["images_tr"],
            labels_tr=paths["labels_tr"],
            modalities=modalities,
            label_file=label_file,
        )
        records.append(rec)
        flag = "+" if rec["status"] == STATUS_OK else "-"
        print(f"  [{flag}] {rec['case_id']}: {rec['status']} {rec['reason']}".rstrip())

    log_df = pd.DataFrame(records)
    log_df.to_csv(paths["output_log"], index=False)

    n_ok = int((log_df["status"] == STATUS_OK).sum()) if not log_df.empty else 0
    n_fail = len(log_df) - n_ok

    write_stage3_dataset_json(
        config,
        paths,
        num_training=n_ok,
        modalities=modalities,
    )

    print(f"[*] Log: {paths['output_log']}")
    print(f"[*] dataset.json: {paths['dataset_json']} (numTraining={n_ok})")
    print(f"[*] Resumen train: {n_ok} {STATUS_OK}, {n_fail} {STATUS_FAIL}")
    return log_df
