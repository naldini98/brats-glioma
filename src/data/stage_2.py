"""Stage 2: Split Manager — train y test con imágenes + labels emparejados.

Entrada: output/stage_1 (SUCCESS_PREPROCESSED).
Salida bajo output/stage_2/splits/:
  train/imagesTr + train/labelsTr   (~85 %, CV la hace nnU-Net en Stage 4)
  test/imagesTs + test/labelsTs     (~15 %, test ciego)
  test_ground_truth/ (copia plana de labels test para eval Stage 4)
  dataset_split.json, processing_log.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.dataset_split import (
    DEFAULT_RATIOS,
    SPLIT_NAMES,
    ground_truth_filename,
    load_split_manifest,
    partition_case_ids,
    split_image_label_dirs,
    verify_all_split_bundles,
    verify_split_counts,
    verify_split_disjoint,
    verify_test_ground_truth_dir,
    write_split_manifest,
)
from src.data.dicom_geometry import ensure_dir
from src.data.nnunet_convert import copy_nifti
from src.data.stage_0 import output_paths_for_case, project_root_from_config
from src.data.stage_1 import STATUS_OK as STAGE1_OK
from src.data.stage_1 import stage1_paths

STATUS_OK = "SUCCESS_SPLIT"
STATUS_FAIL = "FAILED"

# Claves del pipeline antiguo (Stage 2 = convert nnU-Net). No usar.
STAGE2_OBSOLETE_KEYS = frozenset({"nnunet_raw", "conversion_log"})


def validate_stage2_config(cfg: dict) -> None:
    obsolete = STAGE2_OBSOLETE_KEYS & set(cfg.keys())
    if obsolete:
        names = ", ".join(sorted(obsolete))
        raise ValueError(
            f"stage_2 contiene claves obsoletas ({names}). "
            "Stage 2 es solo split train/test. nnU-Net va en stage_3.nnunet_raw. "
            "Editá config.yaml y borrá esas claves."
        )
    ratios = cfg.get("ratios") or {}
    if "val" in ratios:
        raise ValueError(
            "stage_2.ratios.val ya no existe. Usá solo train y test (ej. train: 0.85, test: 0.15)."
        )


def stage2_paths(config: dict) -> dict[str, Path]:
    root = project_root_from_config(config)
    cfg = config["stage_2"]
    validate_stage2_config(cfg)
    s1 = stage1_paths(config)

    def _p(key: str, default: str | None = None) -> Path:
        raw = cfg[key] if key in cfg else default
        if raw is None:
            raise KeyError(key)
        value = Path(raw)
        return value if value.is_absolute() else (root / value)

    splits_base = _p("splits_dir", "output/stage_2/splits")

    splits: dict[str, Path] = {}
    for name in SPLIT_NAMES:
        splits[name] = splits_base / name

    return {
        "root": root,
        "input_log": s1["output_log"],
        "input_volumes_dir": s1["volumes_dir"],
        "input_masks_dir": s1["masks_dir"],
        "output_log": _p("output_log", "output/stage_2/processing_log.csv"),
        "audit_report": _p("audit_report", "output/stage_2/reporte_auditoria_stage_2.md"),
        "split_manifest": _p("split_manifest", "output/stage_2/dataset_split.json"),
        "splits_dir": splits_base,
        "splits": splits,
        "test_ground_truth_dir": _p(
            "test_ground_truth_dir", "output/stage_2/test_ground_truth"
        ),
    }


def load_approved_case_ids(stage1_log: Path) -> list[str]:
    """Casos aprobados por Stage 1."""
    if not stage1_log.exists():
        raise FileNotFoundError(f"No se encontró log Stage 1: {stage1_log}")
    df = pd.read_csv(stage1_log)
    if "case_id" not in df.columns or "status" not in df.columns:
        raise ValueError("preprocessing_log Stage 1 sin columnas case_id/status")
    approved = df[df["status"] == STAGE1_OK]["case_id"].astype(str).tolist()
    return sorted(approved)


def resolve_seed(config: dict) -> int:
    cfg = config.get("stage_2", {})
    if cfg.get("seed") is not None:
        return int(cfg["seed"])
    return int(config.get("project", {}).get("seed", 42))


def resolve_ratios(config: dict) -> dict[str, float]:
    cfg = config.get("stage_2", {})
    validate_stage2_config(cfg)
    ratios = cfg.get("ratios")
    if not ratios:
        return dict(DEFAULT_RATIOS)
    if "train" not in ratios or "test" not in ratios:
        raise ValueError("stage_2.ratios debe definir solo train y test.")
    return {"train": float(ratios["train"]), "test": float(ratios["test"])}


def copy_case_to_split_bundle(
    case_id: str,
    split: str,
    *,
    input_volumes_dir: Path,
    input_masks_dir: Path,
    split_root: Path,
    modalities: list[str],
    label_file: str,
) -> tuple[str, str, dict[str, str]]:
    """Copia modalidades + seg de Stage 1 al bundle images*/labels* del split."""
    dirs = split_image_label_dirs(split_root, split)
    ensure_dir(dirs["images"])
    ensure_dir(dirs["labels"])

    in_paths = output_paths_for_case(
        case_id, input_volumes_dir, input_masks_dir, modalities, label_file
    )
    missing = [k for k, p in in_paths.items() if not p.exists()]
    if missing:
        return STATUS_FAIL, "missing_stage1_input", {"missing": missing}

    out_paths: dict[str, str] = {}
    folder = f"volume_{case_id}"
    out_vol = dirs["images"] / folder
    out_mask = dirs["labels"] / folder
    ensure_dir(out_vol)
    ensure_dir(out_mask)

    for mod in modalities:
        dst = out_vol / f"{folder}-{mod}.nii.gz"
        copy_nifti(in_paths[mod], dst)
        out_paths[mod] = str(dst)

    dst_seg = out_mask / f"{folder}-{label_file}.nii.gz"
    copy_nifti(in_paths[label_file], dst_seg)
    out_paths[label_file] = str(dst_seg)

    return STATUS_OK, "", out_paths


def copy_test_ground_truth_flat(
    case_id: str,
    *,
    labels_ts_dir: Path,
    gt_dir: Path,
    label_file: str,
) -> tuple[str, str, str]:
    """Copia seg del bundle test/labelsTs a test_ground_truth/ (nombre plano)."""
    folder = f"volume_{case_id}"
    src = labels_ts_dir / folder / f"{folder}-{label_file}.nii.gz"
    if not src.exists():
        return STATUS_FAIL, "missing_seg", ""
    dst = gt_dir / ground_truth_filename(case_id)
    copy_nifti(src, dst)
    return STATUS_OK, "", str(dst)


def run_stage_2(config: dict, *, yes: bool = False) -> pd.DataFrame:
    """Particiona casos y materializa bundles imágenes+labels por split."""
    paths = stage2_paths(config)
    cfg = config["stage_2"]
    label_file: str = cfg.get("label_file", "seg")
    modalities: list[str] = list(cfg.get("modalities") or config["stage_1"]["modalities"])
    max_cases = cfg.get("max_cases")
    seed = resolve_seed(config)
    ratios = resolve_ratios(config)

    ensure_dir(paths["splits_dir"])
    for split in SPLIT_NAMES:
        bundle = split_image_label_dirs(paths["splits"][split], split)
        ensure_dir(bundle["images"])
        ensure_dir(bundle["labels"])
    ensure_dir(paths["test_ground_truth_dir"])
    ensure_dir(paths["output_log"].parent)

    case_ids = load_approved_case_ids(paths["input_log"])
    if max_cases is not None:
        case_ids = case_ids[: int(max_cases)]

    print("[*] Stage 2: Split Manager (train + test, imágenes con labels)")
    print(f"    input log:   {paths['input_log']}")
    print(f"    casos:       {len(case_ids)} (SUCCESS_PREPROCESSED)")
    print(f"    seed:        {seed}")
    print(f"    ratios:      {ratios}")
    print(f"    splits dir:  {paths['splits_dir']}")
    for split in SPLIT_NAMES:
        d = split_image_label_dirs(paths["splits"][split], split)
        print(f"    {split}: images={d['images'].name} labels={d['labels'].name}")

    if not case_ids:
        raise ValueError("No hay casos aprobados en Stage 1 para particionar")

    if not yes:
        answer = input("Continuar? [y/N]: ").strip().lower()
        if answer not in {"y", "yes", "s", "si"}:
            print("[-] Cancelado por el usuario.")
            return pd.DataFrame()

    manifest = partition_case_ids(case_ids, ratios=ratios, seed=seed)
    manifest["layout"] = {
        split: {
            "images": str(split_image_label_dirs(paths["splits"][split], split)["images"]),
            "labels": str(split_image_label_dirs(paths["splits"][split], split)["labels"]),
        }
        for split in SPLIT_NAMES
    }
    write_split_manifest(paths["split_manifest"], manifest)

    split_lookup: dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        for cid in manifest[split_name]:
            split_lookup[cid] = split_name

    test_labels_dir = split_image_label_dirs(paths["splits"]["test"], "test")["labels"]

    records: list[dict] = []
    for i, case_id in enumerate(case_ids, start=1):
        split = split_lookup[case_id]
        record: dict = {
            "case_id": case_id,
            "split": split,
            "status": STATUS_OK,
            "reason": "",
            "path_images_dir": "",
            "path_labels_dir": "",
            "path_ground_truth": "",
        }

        status, reason, out_paths = copy_case_to_split_bundle(
            case_id,
            split,
            input_volumes_dir=paths["input_volumes_dir"],
            input_masks_dir=paths["input_masks_dir"],
            split_root=paths["splits"][split],
            modalities=modalities,
            label_file=label_file,
        )
        record["status"] = status
        record["reason"] = reason
        bundle = split_image_label_dirs(paths["splits"][split], split)
        record["path_images_dir"] = str(bundle["images"])
        record["path_labels_dir"] = str(bundle["labels"])
        if status == STATUS_OK and label_file in out_paths:
            record["path_seg"] = out_paths[label_file]

        if split == "test" and status == STATUS_OK:
            gt_status, gt_reason, gt_path = copy_test_ground_truth_flat(
                case_id,
                labels_ts_dir=test_labels_dir,
                gt_dir=paths["test_ground_truth_dir"],
                label_file=label_file,
            )
            if gt_status != STATUS_OK:
                record["status"] = gt_status
                record["reason"] = gt_reason
            record["path_ground_truth"] = gt_path

        records.append(record)
        if i <= 5 or record["status"] != STATUS_OK:
            flag = "+" if record["status"] == STATUS_OK else "-"
            print(f"  [{flag}] {case_id}: {split} {record['reason']}".rstrip())

    log_df = pd.DataFrame(records)
    log_df.to_csv(paths["output_log"], index=False)

    ok_disjoint, msg_disjoint = verify_split_disjoint(manifest)
    ok_counts, msg_counts = verify_split_counts(manifest, len(case_ids))
    ok_gt, msg_gt, _ = verify_test_ground_truth_dir(manifest, paths["test_ground_truth_dir"])
    bundle_rows = verify_all_split_bundles(
        manifest,
        paths["splits"],
        modalities=modalities,
        label_file=label_file,
    )
    bundles_ok = all(r["ok"] for r in bundle_rows)

    print(f"[*] Manifest: {paths['split_manifest']}")
    print(f"[*] counts:   {manifest['counts']}")
    print(f"[*] Log:      {paths['output_log']}")
    gate_ok = ok_disjoint and ok_counts and ok_gt and bundles_ok
    print(
        f"[*] Gates:    disjoint={ok_disjoint} counts={ok_counts} "
        f"bundles={bundles_ok} test_gt={ok_gt}"
        + (f" ({msg_disjoint or msg_counts or msg_gt})" if not gate_ok else "")
    )
    if not gate_ok:
        print("[!] Advertencia: revisar auditoría Stage 2")

    return log_df


def split_source_dirs(config: dict, split: str) -> dict[str, Path]:
    """Directorios images/labels del Stage 2 para un split (entrada Stage 3)."""
    paths = stage2_paths(config)
    if split not in paths["splits"]:
        raise ValueError(f"split desconocido: {split}")
    return split_image_label_dirs(paths["splits"][split], split)


def audit_split_integrity(
    config: dict,
) -> tuple[list[dict], dict[str, Path]]:
    """Checks de integridad para tools/audit_stage_2."""
    paths = stage2_paths(config)
    cfg = config.get("stage_2", {})
    modalities: list[str] = list(cfg.get("modalities") or config["stage_1"]["modalities"])
    label_file: str = cfg.get("label_file", "seg")

    case_ids = load_approved_case_ids(paths["input_log"])
    max_cases = cfg.get("max_cases")
    if max_cases is not None:
        case_ids = case_ids[: int(max_cases)]

    manifest = load_split_manifest(paths["split_manifest"])
    rows: list[dict] = []

    ok_d, msg_d = verify_split_disjoint(manifest)
    rows.append({"check": "disjoint_splits", "ok": ok_d, "detail": msg_d or "OK"})

    ok_c, msg_c = verify_split_counts(manifest, len(case_ids))
    rows.append({"check": "partition_sum", "ok": ok_c, "detail": msg_c or "OK"})

    for bundle in verify_all_split_bundles(
        manifest,
        paths["splits"],
        modalities=modalities,
        label_file=label_file,
    ):
        rows.append(
            {
                "check": bundle["check"],
                "ok": bundle["ok"],
                "detail": bundle["detail"],
                "issues": bundle.get("issues", []),
            }
        )

    ok_gt, msg_gt, issues = verify_test_ground_truth_dir(
        manifest, paths["test_ground_truth_dir"]
    )
    rows.append(
        {
            "check": "test_ground_truth",
            "ok": ok_gt,
            "detail": msg_gt or "OK",
            "issues": issues,
        }
    )

    return rows, paths
