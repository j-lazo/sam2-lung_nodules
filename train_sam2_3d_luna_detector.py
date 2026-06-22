#!/usr/bin/env python3
"""
3D SAM2-feature detector + SAM2 video propagation for LUNA-style nodule segmentation.

Main idea
---------
This script differs from the earlier 2D/2.5D detector script in the detection stage:

Earlier baseline:
    one slice or z-1/z/z+1 pseudo-RGB -> SAM2 image encoder -> 2D detector head

This script:
    N consecutive slices -> SAM2 image encoder per slice -> stack embeddings along z
    -> small 3D CNN detector head -> predicted 3D center + 2D box prompt on central slice
    -> SAM2 video predictor propagates through the volume.

Commands
--------
1) train-feature3d-detector
2) infer-feature3d-detector
3) infer-feature3d-sam2-video

Assumed dataset layout matches your existing sam2_luna_inference.py:
    DATASET_DIR/CT_volumes/*.mhd
    DATASET_DIR/masks_nodules/nifti_data/*mask*contour*nodule*.nii.gz
    DATASET_DIR/annotations.csv
    DATASET_DIR/LUNA16_metadata_split_offical.csv
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import socket
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sam2.build_sam import build_sam2, build_sam2_video_predictor


# =============================================================================
# General utilities
# =============================================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slugify(value: str, max_len: int = 180) -> str:
    value = str(value).replace("+", "plus")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-_.")
    return value[:max_len]


def format_float_for_name(x: Optional[float]) -> str:
    if x is None:
        return "none"
    return f"{x:g}".replace(".", "p")


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def safe_autocast(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[amp_dtype])


def configure_torch(device: torch.device, allow_tf32: bool) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        if hasattr(torch, "set_float32_matmul_precision") and allow_tf32:
            torch.set_float32_matmul_precision("high")


def dice_score(mask1: np.ndarray, mask2: np.ndarray, smooth: float = 1e-6) -> float:
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    inter = np.logical_and(mask1, mask2).sum(dtype=np.float64)
    total = mask1.sum(dtype=np.float64) + mask2.sum(dtype=np.float64)
    return float((2.0 * inter + smooth) / (total + smooth))


def write_pred_volume(pred: np.ndarray, reference_image: sitk.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(pred.astype(np.uint8))
    img.CopyInformation(reference_image)
    sitk.WriteImage(img, str(out_path))



def binary_confusion_counts(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, int, int, int]:
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    tp = int(np.logical_and(gt_b, pred_b).sum())
    fp = int(np.logical_and(~gt_b, pred_b).sum())
    fn = int(np.logical_and(gt_b, ~pred_b).sum())
    tn = int(np.logical_and(~gt_b, ~pred_b).sum())
    return tp, fp, fn, tn


def iou_score(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> float:
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    inter = np.logical_and(gt_b, pred_b).sum(dtype=np.float64)
    union = np.logical_or(gt_b, pred_b).sum(dtype=np.float64)
    return float((inter + smooth) / (union + smooth))


def precision_recall_f1(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> Dict[str, float]:
    tp, fp, fn, tn = binary_confusion_counts(gt, pred)
    precision = float((tp + smooth) / (tp + fp + smooth))
    recall = float((tp + smooth) / (tp + fn + smooth))
    f1 = float((2.0 * precision * recall + smooth) / (precision + recall + smooth))
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, ~eroded)


def surface_distances_mm(mask_a: np.ndarray, mask_b: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> np.ndarray:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if not a.any() or not b.any():
        return np.array([], dtype=np.float32)
    surf_a = _surface(a)
    surf_b = _surface(b)
    dt_b = ndimage.distance_transform_edt(~surf_b, sampling=spacing_zyx)
    dt_a = ndimage.distance_transform_edt(~surf_a, sampling=spacing_zyx)
    d_ab = dt_b[surf_a]
    d_ba = dt_a[surf_b]
    return np.concatenate([d_ab, d_ba]).astype(np.float32)


def hd95_mm(gt: np.ndarray, pred: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> float:
    if not gt.astype(bool).any() and not pred.astype(bool).any():
        return 0.0
    if not gt.astype(bool).any() or not pred.astype(bool).any():
        return float("nan")
    d = surface_distances_mm(gt, pred, spacing_zyx)
    return float(np.percentile(d, 95)) if d.size else float("nan")


def assd_mm(gt: np.ndarray, pred: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> float:
    if not gt.astype(bool).any() and not pred.astype(bool).any():
        return 0.0
    if not gt.astype(bool).any() or not pred.astype(bool).any():
        return float("nan")
    d = surface_distances_mm(gt, pred, spacing_zyx)
    return float(np.mean(d)) if d.size else float("nan")


def volume_similarity(gt: np.ndarray, pred: np.ndarray, smooth: float = 1e-6) -> float:
    g = float(gt.astype(bool).sum())
    p = float(pred.astype(bool).sum())
    return float(1.0 - abs(p - g) / (p + g + smooth))


def get_spacing_zyx(image_itk: sitk.Image) -> Tuple[float, float, float]:
    sx, sy, sz = image_itk.GetSpacing()
    return float(sz), float(sy), float(sx)


def segmentation_metrics(gt: np.ndarray, pred: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> Dict[str, float]:
    pr = precision_recall_f1(gt, pred)
    return {
        "DSC": dice_score(gt, pred),
        "IoU": iou_score(gt, pred),
        "seg_precision": pr["precision"],
        "seg_recall": pr["recall"],
        "seg_f1": pr["f1"],
        "HD95_mm": hd95_mm(gt, pred, spacing_zyx),
        "ASSD_mm": assd_mm(gt, pred, spacing_zyx),
        "volume_similarity": volume_similarity(gt, pred),
        "pred_voxels": int(pred.astype(bool).sum()),
        "gt_voxels": int(gt.astype(bool).sum()),
    }


def extract_gt_nodules_3d(gt_volume: np.ndarray, spacing_zyx: Tuple[float, float, float], min_area: int = 1) -> List[Dict]:
    labeled, num = ndimage.label(gt_volume.astype(bool), structure=ndimage.generate_binary_structure(3, 2))
    nodules: List[Dict] = []
    voxel_volume = float(spacing_zyx[0] * spacing_zyx[1] * spacing_zyx[2])
    for k in range(1, num + 1):
        comp = labeled == k
        vox = int(comp.sum())
        if vox < min_area:
            continue
        coords = np.argwhere(comp)
        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)
        center = coords.mean(axis=0)  # z,y,x in voxels
        extent_vox = np.array([zmax - zmin + 1, ymax - ymin + 1, xmax - xmin + 1], dtype=np.float32)
        extent_mm = extent_vox * np.array(spacing_zyx, dtype=np.float32)
        eq_diam_mm = float((6.0 * vox * voxel_volume / np.pi) ** (1.0 / 3.0))
        nodules.append({
            "nodule_id": int(k),
            "mask": comp.astype(np.uint8),
            "center_zyx": center.astype(np.float32),
            "bbox_zyx": [int(zmin), int(ymin), int(xmin), int(zmax), int(ymax), int(xmax)],
            "gt_volume_voxels": vox,
            "gt_volume_mm3": float(vox * voxel_volume),
            "gt_diameter_vox": float(max(extent_vox)),
            "gt_diameter_mm": float(max(extent_mm)),
            "gt_equivalent_diameter_mm": eq_diam_mm,
        })
    nodules.sort(key=lambda n: n["nodule_id"])
    return nodules


def candidate_fields(c) -> Tuple[float, float, float, float, Optional[np.ndarray]]:
    z = float(getattr(c, "z"))
    y = float(getattr(c, "y"))
    x = float(getattr(c, "x"))
    score = float(getattr(c, "score", np.nan))
    box = getattr(c, "box", None)
    if box is None:
        box = getattr(c, "box_xyxy", None)
    return z, y, x, score, None if box is None else np.asarray(box, dtype=np.float32)


def center_error(c, nodule: Dict, spacing_zyx: Tuple[float, float, float]) -> Tuple[float, float, int]:
    z, y, x, score, _ = candidate_fields(c)
    center = nodule["center_zyx"]
    dz, dy, dx = z - float(center[0]), y - float(center[1]), x - float(center[2])
    err_vox = float(np.sqrt(dz * dz + dy * dy + dx * dx))
    err_mm = float(np.sqrt((dz * spacing_zyx[0]) ** 2 + (dy * spacing_zyx[1]) ** 2 + (dx * spacing_zyx[2]) ** 2))
    return err_vox, err_mm, int(round(abs(dz)))


def match_candidates_to_gt(gt_volume: np.ndarray, candidates: List, spacing_zyx: Tuple[float, float, float], min_area: int = 1) -> Tuple[List[Dict], List[Dict], List[int]]:
    nodules = extract_gt_nodules_3d(gt_volume, spacing_zyx, min_area=min_area)
    matches: List[Dict] = []
    used = set()
    for ni, n in enumerate(nodules):
        best = None
        for ci, c in enumerate(candidates):
            if ci in used:
                continue
            ev, em, ez = center_error(c, n, spacing_zyx)
            radius_vox = max(2.0, 0.5 * float(n["gt_diameter_vox"]))
            radius_mm = max(2.0, 0.5 * float(n["gt_diameter_mm"]))
            if ev <= radius_vox or em <= radius_mm:
                z, y, x, score, box = candidate_fields(c)
                item = {"gt_index": ni, "cand_index": ci, "center_error_vox": ev, "center_error_mm": em, "slice_error": ez, "detector_confidence": score}
                if best is None or item["center_error_mm"] < best["center_error_mm"]:
                    best = item
        if best is not None:
            used.add(best["cand_index"])
            matches.append(best)
    unmatched_gt = [i for i in range(len(nodules)) if i not in {m["gt_index"] for m in matches}]
    unmatched_cand = [i for i in range(len(candidates)) if i not in used]
    return matches, nodules, unmatched_cand


def detection_localization_metrics(gt_volume: np.ndarray, candidates: List, spacing_zyx: Tuple[float, float, float], min_area: int = 1) -> Dict[str, float]:
    matches, nodules, unmatched_cand = match_candidates_to_gt(gt_volume, candidates, spacing_zyx, min_area=min_area)
    tp = len(matches)
    fn = max(0, len(nodules) - tp)
    fp = len(unmatched_cand)
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    center_errs_mm = [m["center_error_mm"] for m in matches]
    center_errs_vox = [m["center_error_vox"] for m in matches]
    slice_errs = [m["slice_error"] for m in matches]
    return {
        "det_TP": tp,
        "det_FP": fp,
        "det_FN": fn,
        "det_precision": precision,
        "det_recall": recall,
        "det_f1": f1,
        "n_gt_nodules": len(nodules),
        "n_candidates": len(candidates),
        "mean_center_error_mm": float(np.mean(center_errs_mm)) if center_errs_mm else np.nan,
        "median_center_error_mm": float(np.median(center_errs_mm)) if center_errs_mm else np.nan,
        "mean_center_error_vox": float(np.mean(center_errs_vox)) if center_errs_vox else np.nan,
        "mean_slice_error": float(np.mean(slice_errs)) if slice_errs else np.nan,
        "best_det_score": float(max([candidate_fields(c)[3] for c in candidates], default=np.nan)),
    }


def per_nodule_metrics(case: CaseData, pred: Optional[np.ndarray], candidates: List, min_area: int = 1) -> List[Dict]:
    spacing_zyx = get_spacing_zyx(case.image_itk)
    matches, nodules, unmatched = match_candidates_to_gt(case.gt_volume, candidates, spacing_zyx, min_area=min_area)
    match_by_gt = {m["gt_index"]: m for m in matches}
    rows: List[Dict] = []
    for i, n in enumerate(nodules):
        gt_mask = n["mask"].astype(np.uint8)
        m = match_by_gt.get(i)
        if pred is None:
            dsc = np.nan
            hd = np.nan
        else:
            dsc = dice_score(gt_mask, pred)
            hd = hd95_mm(gt_mask, pred, spacing_zyx)
        row = {
            "SeriesUID": case.series_id,
            "nodule_id": n["nodule_id"],
            "GT_diameter_vox": n["gt_diameter_vox"],
            "GT_diameter_mm": n["gt_diameter_mm"],
            "GT_equivalent_diameter_mm": n["gt_equivalent_diameter_mm"],
            "GT_volume_voxels": n["gt_volume_voxels"],
            "GT_volume_mm3": n["gt_volume_mm3"],
            "Detector_confidence": m["detector_confidence"] if m else np.nan,
            "Center_error_vox": m["center_error_vox"] if m else np.nan,
            "Center_error_mm": m["center_error_mm"] if m else np.nan,
            "Slice_error": m["slice_error"] if m else np.nan,
            "Dice": dsc,
            "HD95_mm": hd,
            "detected": bool(m is not None),
        }
        rows.append(row)
    return rows


def summarize_detection_operating_points(volume_rows: List[Dict]) -> Dict[str, float]:
    if not volume_rows:
        return {}
    total_tp = int(np.nansum([r.get("det_TP", 0) for r in volume_rows]))
    total_fp = int(np.nansum([r.get("det_FP", 0) for r in volume_rows]))
    total_fn = int(np.nansum([r.get("det_FN", 0) for r in volume_rows]))
    n_scans = max(1, len([r for r in volume_rows if r.get("status") == "ok"]))
    precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) else 0.0
    recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"overall_det_TP": total_tp, "overall_det_FP": total_fp, "overall_det_FN": total_fn, "overall_det_precision": precision, "overall_det_recall": recall, "overall_det_f1": f1, "FP_per_scan": float(total_fp / n_scans)}


def save_detection_summary(rows: List[Dict], out_dir: Path) -> None:
    summary = summarize_detection_operating_points(rows)
    if summary:
        pd.DataFrame([summary]).to_csv(out_dir / "detection_summary.csv", index=False)
        with open(out_dir / "detection_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def plot_training_metrics(metrics_csv: Path, out_dir: Path) -> None:
    if not metrics_csv.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: could not import matplotlib for training plots: {exc}")
        return
    df = pd.read_csv(metrics_csv)
    if df.empty or "epoch" not in df.columns:
        return
    numeric_cols = [c for c in df.columns if c != "epoch" and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return
    loss_cols = [c for c in numeric_cols if "loss" in c.lower()]
    metric_cols = [c for c in numeric_cols if c not in loss_cols and c != "lr"]
    def _plot(cols: List[str], filename: str, title: str):
        if not cols:
            return
        plt.figure(figsize=(10, 6))
        for c in cols:
            plt.plot(df["epoch"], df[c], marker="o", linewidth=1.5, label=c)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()
    _plot(loss_cols, "training_validation_losses.png", "Training/validation losses")
    _plot(metric_cols, "training_validation_metrics.png", "Training/validation metrics")

def stable_uint32_seed(*items) -> int:
    payload = "::".join(str(x) for x in items).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


# =============================================================================
# Experiment directory/config
# =============================================================================


def build_experiment_name(args: argparse.Namespace, command: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = "feat3d"
    n_tag = f"N{getattr(args, 'num_context_slices', 'na')}"
    h_tag = f"hd{getattr(args, 'detector_head_dim', 'na')}"
    lr_tag = f"lr{format_float_for_name(getattr(args, 'lr', None))}"
    bs_tag = f"bs{getattr(args, 'batch_size', getattr(args, 'eval_batch_size', 'na'))}"
    ep_tag = f"ep{getattr(args, 'epochs', 'na')}"
    split_tag = f"v{format_float_for_name(getattr(args, 'val_ratio', None))}_t{format_float_for_name(getattr(args, 'test_ratio', None))}"
    triplet = "triplet" if getattr(args, "use_triplet_channels", False) else "singlech"
    parts = [stamp, command, model_tag, n_tag, h_tag, bs_tag, ep_tag, lr_tag, split_tag, triplet]
    if getattr(args, "max_cases", None) is not None:
        parts.append(f"max{args.max_cases}")
    return slugify("_".join(map(str, parts)))


def make_output_dir(args: argparse.Namespace, command: str) -> Path:
    root = Path(args.output_dir).expanduser().resolve()
    if args.create_experiment_dir:
        exp_name = args.experiment_name or build_experiment_name(args, command)
        out = root / exp_name
    else:
        out = root
    out.mkdir(parents=True, exist_ok=args.overwrite_experiment)
    return out


def save_config(args: argparse.Namespace, out_dir: Path, command: str, index: Optional["DatasetIndex"] = None) -> None:
    cfg = {
        "command": command,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "argv": os.sys.argv,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    if index is not None:
        cfg["dataset"] = {
            "dataset_dir": str(index.dataset_dir),
            "volumes_dir": str(index.volumes_dir),
            "masks_dir": str(index.masks_dir),
            "annotations_csv": str(index.annotations_csv),
            "links_csv": str(index.links_csv),
            "n_selected_volumes": len(index.volume_ids),
        }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


# =============================================================================
# CT normalization and SAM2 frame construction
# =============================================================================


def normalize_ct_to_uint8(img_2d: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    x = img_2d.astype(np.float32)
    x = np.clip(x, hu_min, hu_max)
    x = (x - hu_min) / max(hu_max - hu_min, 1e-6)
    return np.round(x * 255.0).astype(np.uint8)


def build_sam2_input_slice(image_array: np.ndarray, z: int, use_triplet_channels: bool, hu_min: float, hu_max: float) -> np.ndarray:
    if use_triplet_channels:
        z0 = max(z - 1, 0)
        z1 = z
        z2 = min(z + 1, image_array.shape[0] - 1)
        return np.stack([
            normalize_ct_to_uint8(image_array[z0], hu_min, hu_max),
            normalize_ct_to_uint8(image_array[z1], hu_min, hu_max),
            normalize_ct_to_uint8(image_array[z2], hu_min, hu_max),
        ], axis=-1)
    ch = normalize_ct_to_uint8(image_array[z], hu_min, hu_max)
    return np.stack([ch, ch, ch], axis=-1)


def numpy_rgb_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def build_frame_tensor_stack(case: "CaseData", z_start: int, n_slices: int, args: argparse.Namespace) -> torch.Tensor:
    Z = case.image_array.shape[0]
    frames = []
    for k in range(n_slices):
        z = int(np.clip(z_start + k, 0, Z - 1))
        rgb = build_sam2_input_slice(case.image_array, z, args.use_triplet_channels, args.hu_min, args.hu_max)
        frames.append(numpy_rgb_to_tensor(rgb))
    return torch.stack(frames, dim=0)  # N,3,H,W


def write_volume_as_sam2_frames(volume_zyx: np.ndarray, out_dir: Path, use_triplet_channels: bool, hu_min: float, hu_max: float, frame_ext: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for z in range(volume_zyx.shape[0]):
        rgb = build_sam2_input_slice(volume_zyx, z, use_triplet_channels, hu_min, hu_max)
        Image.fromarray(rgb).save(out_dir / f"{z:05d}{frame_ext}")


# =============================================================================
# Dataset indexing/loading
# =============================================================================


@dataclass(frozen=True)
class DatasetIndex:
    dataset_dir: Path
    volumes_dir: Path
    masks_dir: Path
    annotations_csv: Path
    links_csv: Path
    df_annotations: pd.DataFrame
    df_links: pd.DataFrame
    mask_id_to_file: Dict[int, str]
    volume_ids: List[str]


@dataclass(frozen=True)
class CaseData:
    series_id: str
    image_itk: sitk.Image
    image_array: np.ndarray
    gt_volume: np.ndarray
    mask_path: Path


def read_case_list(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    ids = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return [x[:-4] if x.endswith(".mhd") else x for x in ids]


def build_dataset_index(args: argparse.Namespace) -> DatasetIndex:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    volumes_dir = Path(args.volumes_dir).expanduser().resolve() if args.volumes_dir else dataset_dir / "CT_volumes"
    masks_dir = Path(args.masks_dir).expanduser().resolve() if args.masks_dir else dataset_dir / "masks_nodules" / "nifti_data"
    annotations_csv = Path(args.annotations_csv).expanduser().resolve() if args.annotations_csv else dataset_dir / "annotations.csv"
    links_csv = Path(args.links_csv).expanduser().resolve() if args.links_csv else dataset_dir / "LUNA16_metadata_split_offical.csv"

    for p, name in [(dataset_dir, "dataset_dir"), (volumes_dir, "volumes_dir"), (masks_dir, "masks_dir"), (annotations_csv, "annotations_csv"), (links_csv, "links_csv")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} does not exist: {p}")

    df_annotations = pd.read_csv(annotations_csv)
    df_links = pd.read_csv(links_csv)
    if "seriesuid" not in df_annotations.columns:
        raise ValueError(f"{annotations_csv} must contain column 'seriesuid'")
    if not {"SeriesID", "CID"}.issubset(df_links.columns):
        raise ValueError(f"{links_csv} must contain columns 'SeriesID' and 'CID'")

    mask_files = [
        f.name for f in masks_dir.iterdir()
        if f.is_file() and "mask" in f.name and "contour" in f.name and "circle" not in f.name and "nodule" in f.name
    ]
    mask_id_to_file = {}
    for fname in mask_files:
        try:
            mask_id_to_file[int(fname.split("_")[0])] = fname
        except ValueError:
            pass

    volume_ids = sorted(p.stem for p in volumes_dir.glob("*.mhd"))
    if args.only_annotated:
        annotated = set(df_annotations["seriesuid"].astype(str).tolist())
        volume_ids = [v for v in volume_ids if v in annotated]

    explicit_cases = read_case_list(args.case_list)
    if explicit_cases:
        wanted = set(explicit_cases)
        volume_ids = [v for v in volume_ids if v in wanted]

    if args.dataset_fraction is not None:
        if not (0 < args.dataset_fraction <= 1):
            raise ValueError("--dataset-fraction must be in (0,1]")
        n = max(1, int(math.ceil(len(volume_ids) * args.dataset_fraction)))
        volume_ids = volume_ids[:n]

    if args.max_cases is not None:
        volume_ids = volume_ids[: max(0, args.max_cases)]

    return DatasetIndex(dataset_dir, volumes_dir, masks_dir, annotations_csv, links_csv, df_annotations, df_links, mask_id_to_file, volume_ids)


def load_case(index: DatasetIndex, series_id: str) -> Optional[CaseData]:
    image_path = index.volumes_dir / f"{series_id}.mhd"
    if not image_path.is_file():
        return None
    links = index.df_links[index.df_links["SeriesID"].astype(str) == str(series_id)]
    if len(links) == 0:
        return None
    mask_id = int(links["CID"].iloc[0])
    mask_fname = index.mask_id_to_file.get(mask_id)
    if mask_fname is None:
        return None
    mask_path = index.masks_dir / mask_fname
    if not mask_path.is_file():
        return None
    image_itk = sitk.ReadImage(str(image_path))
    image_array = sitk.GetArrayFromImage(image_itk).astype(np.float32)
    mask_itk = sitk.ReadImage(str(mask_path))
    gt_volume = (sitk.GetArrayFromImage(mask_itk) >= 0.5).astype(np.uint8)
    if image_array.shape != gt_volume.shape:
        raise ValueError(f"Shape mismatch for {series_id}: image {image_array.shape}, mask {gt_volume.shape}")
    return CaseData(series_id, image_itk, image_array, gt_volume, mask_path)


# =============================================================================
# Splits
# =============================================================================


@dataclass(frozen=True)
class SplitIds:
    train: List[str]
    val: List[str]
    test: List[str]
    all: List[str]


def write_split_files(splits: SplitIds, out_dir: Path) -> None:
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train", "val", "test", "all"]:
        ids = getattr(splits, name)
        (split_dir / f"{name}.txt").write_text("\n".join(ids) + ("\n" if ids else ""))


def build_splits(index: DatasetIndex, args: argparse.Namespace) -> SplitIds:
    train_list = read_case_list(args.train_case_list)
    val_list = read_case_list(args.val_case_list)
    test_list = read_case_list(args.test_case_list)
    valid = set(index.volume_ids)

    if train_list is not None or val_list is not None or test_list is not None:
        train = [x for x in (train_list or []) if x in valid]
        val = [x for x in (val_list or []) if x in valid]
        test = [x for x in (test_list or []) if x in valid]
        used = set(train) | set(val) | set(test)
        if train_list is None:
            train = [x for x in index.volume_ids if x not in used]
        return SplitIds(train=train, val=val, test=test, all=index.volume_ids)

    ids = list(index.volume_ids)
    rng = np.random.default_rng(args.seed)
    if args.shuffle_splits:
        ids = list(rng.permutation(ids))
    n = len(ids)
    n_test = int(round(n * args.test_ratio))
    n_val = int(round(n * args.val_ratio))
    n_test = min(max(n_test, 0), n)
    n_val = min(max(n_val, 0), n - n_test)
    test = ids[:n_test]
    val = ids[n_test:n_test + n_val]
    train = ids[n_test + n_val:]
    return SplitIds(train=train, val=val, test=test, all=ids)


def select_eval_ids(splits: SplitIds, eval_split: str) -> List[str]:
    if eval_split == "train":
        return splits.train
    if eval_split == "val":
        return splits.val
    if eval_split == "test":
        return splits.test
    if eval_split == "all":
        return splits.all
    raise ValueError(eval_split)


# =============================================================================
# Nodule components and targets
# =============================================================================


@dataclass(frozen=True)
class Nodule3D:
    component_id: int
    center_zyx: Tuple[int, int, int]
    bbox_zyx: Tuple[int, int, int, int, int, int]  # z1,y1,x1,z2,y2,x2 inclusive
    area_voxels: int


def extract_3d_nodules(mask_zyx: np.ndarray, min_voxels: int = 1) -> List[Nodule3D]:
    structure = ndimage.generate_binary_structure(rank=3, connectivity=2)
    labeled, num = ndimage.label(mask_zyx.astype(np.uint8), structure=structure)
    out: List[Nodule3D] = []
    for cid in range(1, num + 1):
        comp = labeled == cid
        area = int(comp.sum())
        if area < min_voxels:
            continue
        zs, ys, xs = np.where(comp)
        z1, z2 = int(zs.min()), int(zs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        dist = ndimage.distance_transform_edt(comp)
        cz, cy, cx = [int(v) for v in np.unravel_index(np.argmax(dist), dist.shape)]
        out.append(Nodule3D(cid, (cz, cy, cx), (z1, y1, x1, z2, y2, x2), area))
    out.sort(key=lambda n: (n.center_zyx[0], n.center_zyx[1], n.center_zyx[2]))
    return out


def expand_3d_bbox_for_prompt(n: Nodule3D, shape_zyx: Tuple[int, int, int], xy_expand: float) -> Tuple[int, int, int, int, int, int]:
    Z, H, W = shape_zyx
    z1, y1, x1, z2, y2, x2 = n.bbox_zyx
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(x2 - x1 + 1.0, 4.0) * xy_expand
    bh = max(y2 - y1 + 1.0, 4.0) * xy_expand
    nx1 = int(max(0, math.floor(cx - bw / 2.0)))
    nx2 = int(min(W - 1, math.ceil(cx + bw / 2.0)))
    ny1 = int(max(0, math.floor(cy - bh / 2.0)))
    ny2 = int(min(H - 1, math.ceil(cy + bh / 2.0)))
    return z1, ny1, nx1, z2, ny2, nx2


def draw_gaussian_3d(hm: np.ndarray, cz: float, cy: float, cx: float, rz: float, ry: float, rx: float) -> None:
    D, H, W = hm.shape
    rz = max(float(rz), 1.0)
    ry = max(float(ry), 1.0)
    rx = max(float(rx), 1.0)
    z0 = max(0, int(round(cz - 3 * rz)))
    z1 = min(D - 1, int(round(cz + 3 * rz)))
    y0 = max(0, int(round(cy - 3 * ry)))
    y1 = min(H - 1, int(round(cy + 3 * ry)))
    x0 = max(0, int(round(cx - 3 * rx)))
    x1 = min(W - 1, int(round(cx + 3 * rx)))
    if z1 < z0 or y1 < y0 or x1 < x0:
        return
    zz = np.arange(z0, z1 + 1, dtype=np.float32)
    yy = np.arange(y0, y1 + 1, dtype=np.float32)
    xx = np.arange(x0, x1 + 1, dtype=np.float32)
    zzz, yyy, xxx = np.meshgrid(zz, yy, xx, indexing="ij")
    g = np.exp(-(((zzz - cz) ** 2) / (2 * rz**2) + ((yyy - cy) ** 2) / (2 * ry**2) + ((xxx - cx) ** 2) / (2 * rx**2)))
    hm[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = np.maximum(hm[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1], g)


def build_3d_targets(
    nodules: List[Nodule3D],
    window_start_z: int,
    image_shape_zyx: Tuple[int, int, int],
    out_dhw: Tuple[int, int, int],
    xy_expand: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build heatmap and [l,t,r,b,front,back] distance targets.

    Distances l/t/r/b are in image pixels; front/back are in slice indices.
    """
    Z, H, W = image_shape_zyx
    D, Oh, Ow = out_dhw
    sx = W / float(Ow)
    sy = H / float(Oh)
    sz = 1.0  # model depth grid corresponds to actual input slices, one per frame

    hm = np.zeros((D, Oh, Ow), dtype=np.float32)
    dist_t = np.zeros((6, D, Oh, Ow), dtype=np.float32)
    dist_m = np.zeros((D, Oh, Ow), dtype=np.float32)

    for n in nodules:
        cz, cy, cx = n.center_zyx
        local_z = cz - window_start_z
        if local_z < 0 or local_z >= D:
            continue
        z1, y1, x1, z2, y2, x2 = expand_3d_bbox_for_prompt(n, image_shape_zyx, xy_expand)
        gcx = cx / sx
        gcy = cy / sy
        gcz = local_z / sz
        rx = max((x2 - x1 + 1) / sx * 0.25, 1.0)
        ry = max((y2 - y1 + 1) / sy * 0.25, 1.0)
        rz = max((z2 - z1 + 1) * 0.25, 1.0)
        draw_gaussian_3d(hm, gcz, gcy, gcx, rz, ry, rx)

        gx = int(np.clip(round(gcx), 0, Ow - 1))
        gy = int(np.clip(round(gcy), 0, Oh - 1))
        gz = int(np.clip(round(gcz), 0, D - 1))
        px = (gx + 0.5) * sx
        py = (gy + 0.5) * sy
        pz = window_start_z + gz
        dist_m[gz, gy, gx] = 1.0
        dist_t[:, gz, gy, gx] = np.array([
            max(px - x1, 0.0),
            max(py - y1, 0.0),
            max(x2 - px, 0.0),
            max(y2 - py, 0.0),
            max(pz - z1, 0.0),
            max(z2 - pz, 0.0),
        ], dtype=np.float32)

    return torch.from_numpy(hm[None]), torch.from_numpy(dist_t), torch.from_numpy(dist_m[None])


# =============================================================================
# Datasets
# =============================================================================


@dataclass(frozen=True)
class WindowSample:
    series_id: str
    z_start: int
    is_positive: bool


class LUNAFeature3DWindowDataset(Dataset):
    def __init__(self, index: DatasetIndex, volume_ids: Sequence[str], args: argparse.Namespace, split: str):
        self.index = index
        self.volume_ids = list(volume_ids)
        self.args = args
        self.split = split
        self.case_cache: Dict[str, CaseData] = {}
        self.samples = self._build_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples built for split={split}")

    def _load_case(self, series_id: str) -> CaseData:
        if self.args.cache_cases and series_id in self.case_cache:
            return self.case_cache[series_id]
        case = load_case(self.index, series_id)
        if case is None:
            raise FileNotFoundError(series_id)
        if self.args.cache_cases:
            self.case_cache[series_id] = case
        return case

    def _positive_window_start(self, center_z: int, Z: int) -> int:
        half = self.args.num_context_slices // 2
        return int(np.clip(center_z - half, 0, max(0, Z - self.args.num_context_slices)))

    def _build_samples(self) -> List[WindowSample]:
        rng = np.random.default_rng(self.args.seed + (0 if self.split == "train" else 10000))
        positives: List[WindowSample] = []
        negatives: List[WindowSample] = []
        print(f"Building {self.split} 3D-window index from {len(self.volume_ids)} volumes...")
        for series_id in tqdm(self.volume_ids, desc=f"index-{self.split}"):
            case = load_case(self.index, series_id)
            if case is None:
                continue
            Z = case.gt_volume.shape[0]
            nodules = extract_3d_nodules(case.gt_volume, self.args.min_component_voxels)
            pos_starts = set()
            positive_z = set()
            for n in nodules:
                z0 = self._positive_window_start(n.center_zyx[0], Z)
                positives.append(WindowSample(series_id, z0, True))
                pos_starts.add(z0)
                z1, _, _, z2, _, _ = n.bbox_zyx
                positive_z.update(range(z1, z2 + 1))

            possible = list(range(0, max(1, Z - self.args.num_context_slices + 1), max(1, self.args.negative_window_stride)))
            neg_pool = []
            for z0 in possible:
                z_range = set(range(z0, min(Z, z0 + self.args.num_context_slices)))
                if len(z_range & positive_z) == 0 and z0 not in pos_starts:
                    neg_pool.append(z0)
            if len(neg_pool) > 0:
                take = min(self.args.max_negative_windows_per_case, len(neg_pool))
                for z0 in rng.choice(neg_pool, size=take, replace=False):
                    negatives.append(WindowSample(series_id, int(z0), False))

        if self.split == "train":
            desired_total = int(round(len(positives) / max(self.args.positive_fraction, 1e-6))) if positives else len(negatives)
            desired_neg = max(0, desired_total - len(positives))
            if len(negatives) > desired_neg:
                idx = rng.choice(len(negatives), size=desired_neg, replace=False)
                negatives = [negatives[i] for i in idx]
            samples = positives + negatives
            rng.shuffle(samples)
        else:
            samples = positives + negatives
        print(f"{self.split}: positive windows={len(positives)}, negative windows={len(negatives)}, total={len(samples)}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        case = self._load_case(s.series_id)
        x = build_frame_tensor_stack(case, s.z_start, self.args.num_context_slices, self.args)
        nodules = extract_3d_nodules(case.gt_volume, self.args.min_component_voxels)
        return {
            "frames": x,  # N,3,H,W
            "series_id": s.series_id,
            "z_start": int(s.z_start),
            "image_shape": torch.tensor(case.gt_volume.shape, dtype=torch.long),
            "nodules": nodules,
            "is_positive": bool(s.is_positive),
        }


def collate_feature3d(batch: List[Dict]) -> Dict:
    return {
        "frames": torch.stack([b["frames"] for b in batch], dim=0),
        "series_id": [b["series_id"] for b in batch],
        "z_start": torch.tensor([b["z_start"] for b in batch], dtype=torch.long),
        "image_shape": torch.stack([b["image_shape"] for b in batch], dim=0),
        "nodules": [b["nodules"] for b in batch],
        "is_positive": torch.tensor([b["is_positive"] for b in batch], dtype=torch.bool),
    }


# =============================================================================
# Model
# =============================================================================


class Conv3DGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SAM2Feature3DDetector(nn.Module):
    """SAM2 slice encoder + 3D CNN detector on stacked SAM2 embeddings.

    Input shape:  B,N,3,H,W
    Output:
        objectness_logits: B,1,N,Hf,Wf
        dist_3d:           B,6,N,Hf,Wf [l,t,r,b,front,back]
    """

    def __init__(self, sam2_model: nn.Module, head_dim: int = 256, freeze_encoder: bool = True, feature_index: int = -1):
        super().__init__()
        self.sam2 = sam2_model
        self.freeze_encoder = freeze_encoder
        self.feature_index = feature_index
        if freeze_encoder:
            for p in self.sam2.parameters():
                p.requires_grad = False
        self.proj = nn.Sequential(
            nn.LazyConv3d(head_dim, kernel_size=1),
            nn.GroupNorm(8, head_dim),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            Conv3DGNAct(head_dim, head_dim),
            Conv3DGNAct(head_dim, head_dim),
            Conv3DGNAct(head_dim, head_dim),
        )
        self.objectness = nn.Conv3d(head_dim, 1, kernel_size=1)
        self.dist_3d = nn.Conv3d(head_dim, 6, kernel_size=1)

    def extract_slice_features(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: B,N,3,H,W -> B,C,N,Hf,Wf
        B, N, C, H, W = frames.shape
        flat = frames.reshape(B * N, C, H, W)
        if self.freeze_encoder:
            with torch.no_grad():
                backbone_out = self.sam2.forward_image(flat)
                _, vision_feats, _, feat_sizes = self.sam2._prepare_backbone_features(backbone_out)
        else:
            backbone_out = self.sam2.forward_image(flat)
            _, vision_feats, _, feat_sizes = self.sam2._prepare_backbone_features(backbone_out)
        feat = vision_feats[self.feature_index]
        h, w = feat_sizes[self.feature_index]
        c = feat.shape[-1]
        feat = feat.permute(1, 2, 0).reshape(B * N, c, h, w).contiguous()
        feat = feat.reshape(B, N, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return feat

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.extract_slice_features(frames)
        y = self.proj(feat)
        y = self.head(y)
        return {
            "objectness_logits": self.objectness(y),
            "dist_3d": F.softplus(self.dist_3d(y)) + 1e-3,
        }


# =============================================================================
# Loss and decoding
# =============================================================================


def focal_heatmap_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    pred = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
    pos = targets.eq(1.0).float()
    neg = targets.lt(1.0).float()
    neg_w = torch.pow(1.0 - targets, beta)
    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos
    neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_w * neg
    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def build_batch_3d_targets(batch: Dict, outputs: Dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device):
    B, _, D, Oh, Ow = outputs["objectness_logits"].shape
    heatmaps, dists, masks = [], [], []
    for i in range(B):
        shape = tuple(int(x) for x in batch["image_shape"][i].tolist())
        z_start = int(batch["z_start"][i].item())
        hm, dist, mask = build_3d_targets(batch["nodules"][i], z_start, shape, (D, Oh, Ow), args.box_expand)
        heatmaps.append(hm)
        dists.append(dist)
        masks.append(mask)
    return torch.stack(heatmaps).to(device), torch.stack(dists).to(device), torch.stack(masks).to(device)


def detector3d_loss(outputs: Dict[str, torch.Tensor], heat_t: torch.Tensor, dist_t: torch.Tensor, dist_m: torch.Tensor, dist_loss_weight: float):
    obj_loss = focal_heatmap_loss(outputs["objectness_logits"], heat_t)
    pred = outputs["dist_3d"]
    if dist_m.sum() > 0:
        m = dist_m.expand_as(pred)
        reg_loss = F.smooth_l1_loss(pred * m, dist_t * m, reduction="sum") / m.sum().clamp(min=1.0)
    else:
        reg_loss = pred.sum() * 0.0
    total = obj_loss + dist_loss_weight * reg_loss
    return total, {"loss": float(total.detach()), "obj_loss": float(obj_loss.detach()), "dist_loss": float(reg_loss.detach())}


@dataclass
class Candidate3D:
    series_id: str
    score: float
    z: int
    x: float
    y: float
    box_xyxy: np.ndarray
    z_range: Tuple[int, int]


def clip_box_xyxy(box: np.ndarray, H: int, W: int) -> np.ndarray:
    b = np.asarray(box, dtype=np.float32).copy()
    b[0] = np.clip(b[0], 0, W - 1)
    b[2] = np.clip(b[2], 0, W - 1)
    b[1] = np.clip(b[1], 0, H - 1)
    b[3] = np.clip(b[3], 0, H - 1)
    if b[2] < b[0]:
        b[0], b[2] = b[2], b[0]
    if b[3] < b[1]:
        b[1], b[3] = b[3], b[1]
    return b


def nms_3d_candidates(cands: List[Candidate3D], z_merge: int, xy_merge: float, max_keep: int) -> List[Candidate3D]:
    kept: List[Candidate3D] = []
    for c in sorted(cands, key=lambda x: x.score, reverse=True):
        duplicate = False
        for k in kept:
            if abs(c.z - k.z) <= z_merge and math.hypot(c.x - k.x, c.y - k.y) <= xy_merge:
                duplicate = True
                break
        if not duplicate:
            kept.append(c)
        if len(kept) >= max_keep:
            break
    return kept


def decode_3d_outputs(outputs: Dict[str, torch.Tensor], series_id: str, z_start: int, image_shape: Tuple[int, int, int], args: argparse.Namespace) -> List[Candidate3D]:
    Z, H, W = image_shape
    obj = torch.sigmoid(outputs["objectness_logits"])[0, 0].detach().float().cpu().numpy()  # D,Oh,Ow
    dist = outputs["dist_3d"][0].detach().float().cpu().numpy()  # 6,D,Oh,Ow
    D, Oh, Ow = obj.shape
    sx = W / float(Ow)
    sy = H / float(Oh)
    flat = obj.reshape(-1)
    idxs = np.where(flat >= args.det_score_thresh)[0]
    if len(idxs) == 0 and args.keep_best_if_empty:
        idxs = np.array([int(flat.argmax())], dtype=np.int64)
    idxs = idxs[np.argsort(flat[idxs])[::-1]][: max(args.topk_per_window * 4, args.topk_per_window)] if len(idxs) > 0 else []
    cands = []
    for idx in idxs:
        dz = int(idx // (Oh * Ow))
        rem = int(idx % (Oh * Ow))
        gy = int(rem // Ow)
        gx = int(rem % Ow)
        score = float(obj[dz, gy, gx])
        z = int(np.clip(z_start + dz, 0, Z - 1))
        x = float((gx + 0.5) * sx)
        y = float((gy + 0.5) * sy)
        l, t, r, b, front, back = dist[:, dz, gy, gx].tolist()
        box = clip_box_xyxy(np.array([x - l, y - t, x + r, y + b], dtype=np.float32), H, W)
        z1 = int(np.clip(round(z - front), 0, Z - 1))
        z2 = int(np.clip(round(z + back), 0, Z - 1))
        if box[2] - box[0] + 1 < 2 or box[3] - box[1] + 1 < 2:
            continue
        cands.append(Candidate3D(series_id, score, z, x, y, box, (min(z1, z2), max(z1, z2))))
    return nms_3d_candidates(cands, args.video_prompt_z_merge_window, args.video_prompt_xy_merge_dist, args.topk_per_window)


# =============================================================================
# Training/inference helpers
# =============================================================================


def build_model(args: argparse.Namespace, device: torch.device) -> SAM2Feature3DDetector:
    sam2 = build_sam2(args.model_cfg, args.checkpoint, device=device)
    return SAM2Feature3DDetector(
        sam2_model=sam2,
        head_dim=args.detector_head_dim,
        freeze_encoder=not args.unfreeze_encoder,
        feature_index=args.sam2_feature_index,
    ).to(device)


def save_checkpoint(path: Path, model: SAM2Feature3DDetector, optimizer: Optional[torch.optim.Optimizer], epoch: int, best_val: float, args: argparse.Namespace) -> None:
    payload = {
        "epoch": epoch,
        "best_val": best_val,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(model: SAM2Feature3DDetector, path: str, device: torch.device) -> Dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    return ckpt


def run_epoch(model, loader, optimizer, scaler, device, args, desc: str) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "obj_loss": 0.0, "dist_loss": 0.0}
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        frames = batch["frames"].to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train), safe_autocast(device, args.amp_dtype):
            outputs = model(frames)
            heat_t, dist_t, dist_m = build_batch_3d_targets(batch, outputs, args, device)
            loss, logs = detector3d_loss(outputs, heat_t, dist_t, dist_m, args.dist_loss_weight)
        if is_train:
            if scaler is not None and args.amp_dtype == "fp16":
                scaler.scale(loss).backward()
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
        bs = frames.shape[0]
        for k in totals:
            totals[k] += logs[k] * bs
        n += bs
        pbar.set_postfix({k: f"{totals[k] / max(n,1):.4f}" for k in totals})
    return {k: totals[k] / max(n, 1) for k in totals}


def train_feature3d_detector(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    configure_torch(device, args.allow_tf32)
    index = build_dataset_index(args)
    out_dir = make_output_dir(args, "train-feature3d-detector")
    save_config(args, out_dir, "train-feature3d-detector", index)
    splits = build_splits(index, args)
    write_split_files(splits, out_dir)
    print(f"Train/val/test volumes: {len(splits.train)}/{len(splits.val)}/{len(splits.test)}")
    print(f"Experiment directory: {out_dir}")

    train_ds = LUNAFeature3DWindowDataset(index, splits.train, args, "train")
    val_ds = LUNAFeature3DWindowDataset(index, splits.val if splits.val else splits.train, args, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_feature3d)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_feature3d)

    model = build_model(args, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp_dtype == "fp16"))

    rows = []
    best_val = float("inf")
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, scaler, device, args, f"train {epoch}")
        va = run_epoch(model, val_loader, None, None, device, args, f"val {epoch}")
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(rows, f, indent=2)
        save_checkpoint(out_dir / "last_detector.pt", model, optimizer, epoch, best_val, args)
        if va["loss"] < best_val - args.min_delta:
            best_val = va["loss"]
            bad_epochs = 0
            save_checkpoint(out_dir / "best_detector.pt", model, optimizer, epoch, best_val, args)
            print(f"Epoch {epoch}: new best val loss {best_val:.6f}")
        else:
            bad_epochs += 1
            print(f"Epoch {epoch}: val loss {va['loss']:.6f}; bad_epochs={bad_epochs}/{args.patience}")
        if args.patience > 0 and bad_epochs >= args.patience:
            print("Early stopping.")
            break
    plot_training_metrics(out_dir / "metrics.csv", out_dir)
    print(f"Saved best checkpoint: {out_dir / 'best_detector.pt'}")


def make_windows_for_case(case: CaseData, args: argparse.Namespace) -> List[int]:
    Z = case.gt_volume.shape[0]
    max_start = max(0, Z - args.num_context_slices)
    starts = list(range(0, max_start + 1, max(1, args.infer_window_stride)))
    if not starts or starts[-1] != max_start:
        starts.append(max_start)
    return sorted(set(starts))


def run_detector_on_case(model: SAM2Feature3DDetector, case: CaseData, args: argparse.Namespace, device: torch.device) -> List[Candidate3D]:
    model.eval()
    all_cands: List[Candidate3D] = []
    starts = make_windows_for_case(case, args)
    with torch.inference_mode():
        for z0 in starts:
            frames = build_frame_tensor_stack(case, z0, args.num_context_slices, args).unsqueeze(0).to(device)
            with safe_autocast(device, args.amp_dtype):
                outputs = model(frames)
            cands = decode_3d_outputs(outputs, case.series_id, z0, case.gt_volume.shape, args)
            all_cands.extend(cands)
    all_cands = nms_3d_candidates(all_cands, args.video_prompt_z_merge_window, args.video_prompt_xy_merge_dist, args.max_candidates_per_volume)
    return all_cands


def infer_feature3d_detector(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    configure_torch(device, args.allow_tf32)
    index = build_dataset_index(args)
    out_dir = make_output_dir(args, "infer-feature3d-detector")
    save_config(args, out_dir, "infer-feature3d-detector", index)
    splits = build_splits(index, args)
    eval_ids = select_eval_ids(splits, args.eval_split)
    print(f"Evaluating split={args.eval_split}; n={len(eval_ids)}; out={out_dir}")

    model = build_model(args, device)
    load_checkpoint(model, args.detector_checkpoint, device)
    rows = []
    summary = []
    for sid in tqdm(eval_ids, desc="cases"):
        try:
            case = load_case(index, sid)
            if case is None:
                summary.append({"VolumeID": sid, "status": "missing"})
                continue
            cands = run_detector_on_case(model, case, args, device)
            for rank, c in enumerate(cands, start=1):
                rows.append({
                    "VolumeID": sid, "rank": rank, "score": c.score, "z": c.z, "x": c.x, "y": c.y,
                    "x1": float(c.box_xyxy[0]), "y1": float(c.box_xyxy[1]), "x2": float(c.box_xyxy[2]), "y2": float(c.box_xyxy[3]),
                    "z1": c.z_range[0], "z2": c.z_range[1],
                })
            spacing_zyx = get_spacing_zyx(case.image_itk)
            det_metrics = detection_localization_metrics(case.gt_volume, cands, spacing_zyx, min_area=args.min_component_area)
            summary.append({"VolumeID": sid, "status": "ok", **det_metrics})
        except Exception as exc:
            if args.fail_fast:
                raise
            summary.append({"VolumeID": sid, "status": "error", "error": repr(exc)})
            print(f"ERROR {sid}: {repr(exc)}")
        pd.DataFrame(rows).to_csv(out_dir / "candidates.csv", index=False)
        pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    save_detection_summary(summary, out_dir)
    print(f"Saved: {out_dir / 'candidates.csv'}")


def add_sam2_video_prompt(predictor, inference_state, cand: Candidate3D, obj_id: int, prompt_mode: str):
    point = np.array([[cand.x, cand.y]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    box = cand.box_xyxy.astype(np.float32)
    if prompt_mode == "point":
        return predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=cand.z, obj_id=obj_id, points=point, labels=labels)
    if prompt_mode == "box":
        return predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=cand.z, obj_id=obj_id, box=box)
    if prompt_mode == "point+box":
        return predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=cand.z, obj_id=obj_id, points=point, labels=labels, box=box)
    raise ValueError(prompt_mode)


def accumulate_video_propagation(predictor, inference_state, pred_by_obj: Dict[int, np.ndarray], reverse: bool) -> None:
    kwargs = {"reverse": True} if reverse else {}
    try:
        iterator = predictor.propagate_in_video(inference_state, **kwargs)
    except TypeError:
        if reverse:
            print("WARNING: predictor does not support reverse=True; skipping reverse propagation.")
            return
        iterator = predictor.propagate_in_video(inference_state)
    for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
        for i, obj_id in enumerate(out_obj_ids):
            mask_np = out_mask_logits[i].squeeze().float().cpu().numpy()
            pred_by_obj[int(obj_id)][int(out_frame_idx)] |= (mask_np > 0.0).astype(np.uint8)


def run_sam2_video_on_case(video_predictor, case: CaseData, cands: List[Candidate3D], args: argparse.Namespace, device: torch.device, out_dir: Path) -> Tuple[np.ndarray, List[Dict]]:
    Z, H, W = case.gt_volume.shape
    selected = cands[: args.max_video_prompts]
    pred_volume = np.zeros((Z, H, W), dtype=np.uint8)
    logs = []
    if len(selected) == 0:
        return pred_volume, logs

    temp_parent = Path(args.frame_tmp_dir or os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir())
    temp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_{case.series_id}_", dir=str(temp_parent)))
    try:
        write_volume_as_sam2_frames(case.image_array, temp_dir, args.use_triplet_channels, args.hu_min, args.hu_max, args.frame_ext)
        with torch.inference_mode(), safe_autocast(device, args.amp_dtype):
            inference_state = video_predictor.init_state(video_path=str(temp_dir))
            if hasattr(video_predictor, "reset_state"):
                video_predictor.reset_state(inference_state)
            pred_by_obj = {}
            for obj_idx, c in enumerate(selected, start=1):
                add_sam2_video_prompt(video_predictor, inference_state, c, obj_idx, args.sam2_prompt_mode)
                pred_by_obj[obj_idx] = np.zeros((Z, H, W), dtype=np.uint8)
                logs.append({
                    "VolumeID": case.series_id, "obj_id": obj_idx, "det_score": c.score, "z": c.z,
                    "x": c.x, "y": c.y, "x1": float(c.box_xyxy[0]), "y1": float(c.box_xyxy[1]),
                    "x2": float(c.box_xyxy[2]), "y2": float(c.box_xyxy[3]), "z1": c.z_range[0], "z2": c.z_range[1],
                })
            accumulate_video_propagation(video_predictor, inference_state, pred_by_obj, reverse=False)
            if args.video_bidirectional:
                accumulate_video_propagation(video_predictor, inference_state, pred_by_obj, reverse=True)
        for obj_mask in pred_by_obj.values():
            pred_volume |= obj_mask.astype(np.uint8)
        return pred_volume, logs
    finally:
        if args.cleanup_frames:
            shutil.rmtree(temp_dir, ignore_errors=True)


def infer_feature3d_sam2_video(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    configure_torch(device, args.allow_tf32)
    index = build_dataset_index(args)
    out_dir = make_output_dir(args, "infer-feature3d-sam2-video")
    save_config(args, out_dir, "infer-feature3d-sam2-video", index)
    splits = build_splits(index, args)
    eval_ids = select_eval_ids(splits, args.eval_split)
    print(f"Evaluating split={args.eval_split}; n={len(eval_ids)}; out={out_dir}")

    detector = build_model(args, device)
    load_checkpoint(detector, args.detector_checkpoint, device)
    detector.eval()
    video_predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint, device=device, vos_optimized=args.vos_optimized)

    rows = []
    prompt_rows = []
    nodule_rows = []
    pred_root = out_dir / "predicted_volumes"
    for sid in tqdm(eval_ids, desc="cases"):
        try:
            case = load_case(index, sid)
            if case is None:
                rows.append({"VolumeID": sid, "status": "missing"})
                continue
            cands = run_detector_on_case(detector, case, args, device)
            pred, logs = run_sam2_video_on_case(video_predictor, case, cands, args, device, out_dir)
            spacing_zyx = get_spacing_zyx(case.image_itk)
            rows.append({
                "VolumeID": sid,
                "status": "ok",
                "n_video_prompts": min(len(cands), args.max_video_prompts),
                **detection_localization_metrics(case.gt_volume, cands, spacing_zyx, min_area=args.min_component_area),
                **segmentation_metrics(case.gt_volume, pred, spacing_zyx),
            })
            prompt_rows.extend(logs)
            nodule_rows.extend(per_nodule_metrics(case, pred, cands, min_area=args.min_component_area))
            if args.save_volumes:
                write_pred_volume(pred, case.image_itk, pred_root / "feature3d_sam2_video" / f"{sid}_feature3d_sam2_video.nii.gz")
                if args.save_gt_volume:
                    write_pred_volume(case.gt_volume, case.image_itk, pred_root / "gt" / f"{sid}_gt.nii.gz")
        except Exception as exc:
            if args.fail_fast:
                raise
            rows.append({"VolumeID": sid, "status": "error", "error": repr(exc)})
            print(f"ERROR {sid}: {repr(exc)}")
        pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
        if prompt_rows:
            pd.DataFrame(prompt_rows).to_csv(out_dir / "video_prompt_logs.csv", index=False)
        if nodule_rows:
            pd.DataFrame(nodule_rows).to_csv(out_dir / "nodule_metrics.csv", index=False)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    if prompt_rows:
        pd.DataFrame(prompt_rows).to_csv(out_dir / "video_prompt_logs.csv", index=False)
    if nodule_rows:
        pd.DataFrame(nodule_rows).to_csv(out_dir / "nodule_metrics.csv", index=False)
    save_detection_summary(rows, out_dir)
    ok = df[df["status"] == "ok"]
    if len(ok) > 0:
        print(f"Mean DSC: {pd.to_numeric(ok['DSC'], errors='coerce').mean():.6f}")
    print(f"Saved metrics: {out_dir / 'metrics.csv'}")


# =============================================================================
# CLI
# =============================================================================


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--volumes-dir", default=None)
    p.add_argument("--masks-dir", default=None)
    p.add_argument("--annotations-csv", default=None)
    p.add_argument("--links-csv", default=None)
    p.add_argument("--case-list", default=None)
    p.add_argument("--only-annotated", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dataset-fraction", type=float, default=None)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--train-case-list", default=None)
    p.add_argument("--val-case-list", default=None)
    p.add_argument("--test-case-list", default=None)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--shuffle-splits", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--model-cfg", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--create-experiment-dir", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--overwrite-experiment", action="store_true")

    p.add_argument("--use-triplet-channels", action="store_true")
    p.add_argument("--hu-min", type=float, default=-1000.0)
    p.add_argument("--hu-max", type=float, default=400.0)
    p.add_argument("--num-context-slices", type=int, default=9, help="N consecutive slices encoded by SAM2 and stacked as a feature volume.")
    p.add_argument("--sam2-feature-index", type=int, default=-1, help="Which SAM2 vision feature to use; -1 is lowest-res/deepest.")
    p.add_argument("--detector-head-dim", type=int, default=256)
    p.add_argument("--unfreeze-encoder", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--min-component-voxels", type=int, default=1)
    p.add_argument("--box-expand", type=float, default=1.5)


def add_train_args(p: argparse.ArgumentParser) -> None:
    add_common_args(p)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1, help="Usually 1-2 because each sample contains N SAM2-encoded slices.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--positive-fraction", type=float, default=0.50)
    p.add_argument("--max-negative-windows-per-case", type=int, default=8)
    p.add_argument("--negative-window-stride", type=int, default=4)
    p.add_argument("--dist-loss-weight", type=float, default=0.25)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--cache-cases", action="store_true")


def add_infer_args(p: argparse.ArgumentParser) -> None:
    add_common_args(p)
    p.add_argument("--detector-checkpoint", required=True)
    p.add_argument("--eval-split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--infer-window-stride", type=int, default=4)
    p.add_argument("--det-score-thresh", type=float, default=0.20)
    p.add_argument("--topk-per-window", type=int, default=3)
    p.add_argument("--max-candidates-per-volume", type=int, default=20)
    p.add_argument("--video-prompt-z-merge-window", type=int, default=3)
    p.add_argument("--video-prompt-xy-merge-dist", type=float, default=12.0)
    p.add_argument("--keep-best-if-empty", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fail-fast", action="store_true")


def add_video_args(p: argparse.ArgumentParser) -> None:
    add_infer_args(p)
    p.add_argument("--max-video-prompts", type=int, default=5)
    p.add_argument("--sam2-prompt-mode", choices=["point", "box", "point+box"], default="point+box")
    p.add_argument("--video-bidirectional", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vos-optimized", action="store_true")
    p.add_argument("--frame-tmp-dir", default=None)
    p.add_argument("--frame-ext", choices=[".jpg", ".png"], default=".jpg")
    p.add_argument("--cleanup-frames", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-volumes", action="store_true")
    p.add_argument("--save-gt-volume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3D SAM2-feature detector + SAM2 video propagation for LUNA segmentation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train-feature3d-detector", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_train_args(p_train)

    p_det = sub.add_parser("infer-feature3d-detector", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_infer_args(p_det)

    p_vid = sub.add_parser("infer-feature3d-sam2-video", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_video_args(p_vid)
    return parser


def validate_args(args: argparse.Namespace):
    if args.num_context_slices < 1:
        raise ValueError("--num-context-slices must be >= 1")
    if args.num_context_slices % 2 == 0:
        print("WARNING: even --num-context-slices is allowed, but odd values make the central slice interpretation cleaner.")
    if not (0 <= args.val_ratio < 1) or not (0 <= args.test_ratio < 1) or args.val_ratio + args.test_ratio >= 1:
        if not (args.train_case_list or args.val_case_list or args.test_case_list):
            raise ValueError("Need 0 <= val_ratio, test_ratio and val_ratio + test_ratio < 1")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    if args.command == "train-feature3d-detector":
        train_feature3d_detector(args)
    elif args.command == "infer-feature3d-detector":
        infer_feature3d_detector(args)
    elif args.command == "infer-feature3d-sam2-video":
        infer_feature3d_sam2_video(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()