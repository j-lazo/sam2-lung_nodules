#!/usr/bin/env python3
"""
Unified SAM2 inference/evaluation for LUNA-style CT volumes.

Modes
-----
1) video        : Treat a CT volume as a video and propagate prompted masks in 3D.
2) image-single : Slice-wise SAM2 image predictor with single mask output.
3) image-multi  : Slice-wise SAM2 image predictor with SAM2 multimask output.
4) auto         : Slice-wise SAM2 automatic mask generator only.

The script writes one CSV with Dice scores per volume. Optionally, it also writes
predicted volumes as NIfTI files. It supports dataset subsampling and SLURM-style
sharding so multiple H200 jobs can split the dataset without duplicating work. It can also prefetch CT volumes on CPU while the GPU is running inference.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from datetime import datetime
import socket
import re
import hashlib
import yaml

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import tqdm
from PIL import Image
from scipy import ndimage

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


PROMPT_MODES = ("point", "box", "point+box")
IMAGE_MODES = ("auto", "point", "box", "point+box")


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def slugify(value: str, max_len: int = 120):
    """
    Convert arbitrary text into a filesystem-safe short name.
    """
    value = str(value)
    value = value.replace("+", "plus")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-_.")
    return value[:max_len] if len(value) > max_len else value


def format_float_for_name(value: Optional[float]):
    """
    0.10 -> 0p1, 0.005 -> 0p005, None -> all
    """
    if value is None:
        return "all"
    txt = f"{value:g}"
    return txt.replace(".", "p")


def build_experiment_name(args: argparse.Namespace):
    """
    Build a compact experiment folder name from main hyperparameters.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode == "video":
        output_tag = "pm-" + "-".join(m.replace("+", "plus") for m in args.prompt_modes)
    else:
        output_tag = "out-" + "-".join(m.replace("+", "plus") for m in args.image_outputs)

    triplet_tag = "triplet" if args.use_triplet_channels else "singlech"
    frac_tag = f"frac-{format_float_for_name(args.dataset_fraction)}"
    perturb_tag = None
    if getattr(args, "prompt_perturb_rmax", 0) > 0:
        perturb_tag = f"perturb-r{int(args.prompt_perturb_rmax)}"

    box_size_tag = None
    if getattr(args, "box_size_perturb_px", 0) != 0:
        box_size_tag = (
            f"boxsize-{int(args.box_size_perturb_px):+d}px"
            f"-min{int(args.box_min_size_px)}"
        )

    z_perturb_tag = None
    if args.mode == "video" and getattr(args, "video_z_perturb_rmax", 0) > 0:
        z_perturb_tag = f"zperturb-r{int(args.video_z_perturb_rmax)}"
        if getattr(args, "video_z_perturb_only", False):
            z_perturb_tag += "-only"

    shard_tag = None
    if getattr(args, "shard_count", 1) > 1:
        shard_tag = f"shard-{args.shard_index}of{args.shard_count}"

    auto_tag = None
    if args.mode in {"auto", "image-single"} and "auto" in args.image_outputs:
        auto_tag = (
            f"autoA{args.auto_min_area}-{args.auto_max_area}"
            f"_circ{format_float_for_name(args.auto_min_circularity)}"
            f"_sol{format_float_for_name(args.auto_min_solidity)}"
        )

    parts = [
        timestamp,
        args.run_name,
        args.mode,
        output_tag,
        args.amp_dtype,
        triplet_tag,
        frac_tag,
    ]

    if args.max_cases is not None:
        parts.append(f"max-{args.max_cases}")
    if perturb_tag is not None:
        parts.append(perturb_tag)
    if box_size_tag is not None:
        parts.append(box_size_tag)
    if z_perturb_tag is not None:
        parts.append(z_perturb_tag)
    if shard_tag is not None:
        parts.append(shard_tag)
    if auto_tag is not None:
        parts.append(auto_tag)

    return slugify("_".join(parts), max_len=180)


def namespace_to_yaml_dict(args: argparse.Namespace) :
    """
    Convert argparse Namespace to a YAML-safe dictionary.
    Paths are stored as strings.
    """
    cfg = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            cfg[key] = str(value)
        elif isinstance(value, (list, tuple)):
            cfg[key] = list(value)
        else:
            cfg[key] = value
    return cfg


def write_yaml_config(config_path: Path, config: Dict):
    """
    Write config.yaml. Requires PyYAML; falls back to JSON-formatted text
    with .yaml extension if PyYAML is unavailable.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if yaml is not None:
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    else:
        print("WARNING: PyYAML is not installed. Writing JSON-style config to .yaml file.")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)


def setup_experiment_dir(args: argparse.Namespace):
    """
    Create and return the experiment directory.

    If --create-experiment-dir is enabled:
        output-dir / experiment-name
    Else:
        output-dir
    """
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.create_experiment_dir:
        exp_name = args.experiment_name or build_experiment_name(args)
        exp_dir = output_root / exp_name
    else:
        exp_dir = output_root

    exp_dir.mkdir(parents=True, exist_ok=args.overwrite_experiment)

    if args.save_volumes:
        (exp_dir / "predicted_volumes").mkdir(parents=True, exist_ok=True)

    return exp_dir


def save_experiment_config(
    args: argparse.Namespace,
    exp_dir: Path,
    device: torch.device,
    index: Optional[DatasetIndex] = None,):

    """
    Save hyperparameters and useful run metadata.
    """
    config = {
        "experiment": {
            "experiment_dir": str(exp_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "hostname": socket.gethostname(),
            "command": " ".join(os.sys.argv),
        },
        "runtime": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "arguments": namespace_to_yaml_dict(args),
    }

    if index is not None:
        config["dataset_index"] = {
            "dataset_dir": str(index.dataset_dir),
            "volumes_dir": str(index.volumes_dir),
            "masks_dir": str(index.masks_dir),
            "annotations_csv": str(index.annotations_csv),
            "links_csv": str(index.links_csv),
            "n_selected_volumes": len(index.volume_ids),
            "selected_volume_ids": index.volume_ids,
        }

    write_yaml_config(exp_dir / "config.yaml", config)

def normalize_to_uint8(img_2d: np.ndarray):
    img_2d = img_2d.astype(np.float32)
    img_2d -= float(img_2d.min())
    max_val = float(img_2d.max())
    if max_val > 0:
        img_2d /= max_val
    return (img_2d * 255).astype(np.uint8)


def dice_score(mask1: np.ndarray, mask2: np.ndarray, smooth: float = 1e-6):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum(dtype=np.float64)
    total = mask1.sum(dtype=np.float64) + mask2.sum(dtype=np.float64)
    return float((2.0 * intersection + smooth) / (total + smooth))


def build_sam2_input_slice(image_array: np.ndarray, z: int, use_triplet_channels: bool = False):
    """Return [H, W, 3] uint8. If triplet, channels are z-1/z/z+1."""
    if use_triplet_channels:
        z_prev = max(z - 1, 0)
        z_next = min(z + 1, image_array.shape[0] - 1)
        return np.stack(
            [
                normalize_to_uint8(image_array[z_prev]),
                normalize_to_uint8(image_array[z]),
                normalize_to_uint8(image_array[z_next]),
            ],
            axis=-1,
        )
    img = normalize_to_uint8(image_array[z])
    return np.stack([img, img, img], axis=-1)


def safe_autocast(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp_dtype]
    return torch.autocast("cuda", dtype=dtype)


def configure_torch(device: torch.device, amp_dtype: str, allow_tf32: bool):
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        torch.set_grad_enabled(False)
        # H200 benefits from TF32 for fp32 ops and bf16 autocast for SAM2 inference.
        if hasattr(torch, "set_float32_matmul_precision") and allow_tf32:
            torch.set_float32_matmul_precision("high")
    elif device.type == "mps":
        print("WARNING: MPS support for SAM2 may give different or degraded results.")


def select_device(device_arg: str):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def env_int(name: str, default: int):
    """Read an integer environment variable safely. Empty/non-integer values use default."""
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(f"WARNING: ignoring non-integer ${name}={value!r}; using {default}.")
        return default


def stable_uint32_seed(*items) -> int:
    """
    Build a process-independent uint32 seed from stable identifiers.

    Do not use Python's built-in hash() here because it is randomized per process.
    This makes prompt perturbations reproducible across reruns, SLURM shards,
    multiprocessing, and different GPU ranks, as long as the inputs are the same.
    """
    payload = "::".join(str(item) for item in items).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


def sample_prompt_shift_xy(
    base_seed: int,
    series_id: str,
    prompt_scope: str,
    z: int,
    prompt_index: int,
    rmax: int,
) -> Tuple[int, int]:
    """
    Deterministically sample an integer pixel shift (dx, dy) in [-rmax, +rmax].

    The sample depends only on the stable prompt identity, not on iteration order.
    Therefore the same case receives the same perturbations in single-process,
    multi-process, or SLURM-array executions.
    """
    rmax = int(rmax)
    if rmax <= 0:
        return 0, 0
    seed = stable_uint32_seed(base_seed, series_id, prompt_scope, z, prompt_index, rmax)
    rng = np.random.default_rng(seed)
    dx, dy = rng.integers(-rmax, rmax + 1, size=2, endpoint=False)
    return int(dx), int(dy)




def sample_prompt_shift_z(
    base_seed: int,
    series_id: str,
    prompt_scope: str,
    z: int,
    prompt_index: int,
    rmax: int,
) -> int:
    """
    Deterministically sample an integer slice/frame shift dz in [-rmax, +rmax].

    This is used only for video/volume prompts. The sampled z is clipped later to
    [0, Z - 1], so prompts never point outside the volume.
    """
    rmax = int(rmax)
    if rmax <= 0:
        return 0
    seed = stable_uint32_seed(base_seed, series_id, prompt_scope, z, prompt_index, rmax)
    rng = np.random.default_rng(seed)
    return int(rng.integers(-rmax, rmax + 1, endpoint=False))


def clip_z_index(z: int, volume_depth: int) -> int:
    """Clip a z/frame index to valid volume coordinates."""
    return int(np.clip(int(z), 0, int(volume_depth) - 1))


def clip_point_xy(point_xy: np.ndarray, image_shape_hw: Tuple[int, int]) -> np.ndarray:
    """Clip point coordinates to valid image coordinates."""
    H, W = image_shape_hw
    point = np.asarray(point_xy, dtype=np.float32).copy()
    point[..., 0] = np.clip(point[..., 0], 0, W - 1)
    point[..., 1] = np.clip(point[..., 1], 0, H - 1)
    return point


def clip_box_xyxy(box_xyxy: np.ndarray, image_shape_hw: Tuple[int, int]) -> np.ndarray:
    """Clip an xyxy box to valid image coordinates."""
    H, W = image_shape_hw
    box = np.asarray(box_xyxy, dtype=np.float32).copy()
    box[..., 0] = np.clip(box[..., 0], 0, W - 1)
    box[..., 2] = np.clip(box[..., 2], 0, W - 1)
    box[..., 1] = np.clip(box[..., 1], 0, H - 1)
    box[..., 3] = np.clip(box[..., 3], 0, H - 1)
    return box


def ensure_min_box_size_xyxy(
    box_xyxy: Sequence[float],
    image_shape_hw: Tuple[int, int],
    min_size_px: int = 3,
) -> np.ndarray:
    """
    Keep an xyxy box valid and enforce a minimum inclusive width/height.

    For example, min_size_px=3 means x2 - x1 + 1 >= 3 and
    y2 - y1 + 1 >= 3 whenever the image dimensions allow it.
    """
    H, W = image_shape_hw
    min_size_px = max(1, int(min_size_px))
    min_w = min(min_size_px, int(W))
    min_h = min(min_size_px, int(H))

    x1, y1, x2, y2 = clip_box_xyxy(
        np.asarray(box_xyxy, dtype=np.float32).reshape(4),
        image_shape_hw,
    ).tolist()

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if (x2 - x1 + 1.0) < min_w:
        cx = 0.5 * (x1 + x2)
        half = 0.5 * (min_w - 1)
        x1 = cx - half
        x2 = cx + half
        if x1 < 0:
            x2 -= x1
            x1 = 0.0
        if x2 > W - 1:
            x1 -= x2 - (W - 1)
            x2 = float(W - 1)
        x1 = max(0.0, x1)

    if (y2 - y1 + 1.0) < min_h:
        cy = 0.5 * (y1 + y2)
        half = 0.5 * (min_h - 1)
        y1 = cy - half
        y2 = cy + half
        if y1 < 0:
            y2 -= y1
            y1 = 0.0
        if y2 > H - 1:
            y1 -= y2 - (H - 1)
            y2 = float(H - 1)
        y1 = max(0.0, y1)

    return clip_box_xyxy(np.array([x1, y1, x2, y2], dtype=np.float32), image_shape_hw)


def resize_box_xyxy(
    box_xyxy: Sequence[float],
    image_shape_hw: Tuple[int, int],
    delta_px: int = 0,
    min_size_px: int = 3,
) -> np.ndarray:
    """
    Resize a prompt box by delta_px in every direction.

    delta_px = 0 keeps the original box.
    delta_px > 0 expands the box.
    delta_px < 0 shrinks the box.

    The returned box is clipped to the image and kept at least min_size_px
    pixels wide/high whenever possible.
    """
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float32).reshape(4).tolist()
    d = int(delta_px)
    resized = np.array([x1 - d, y1 - d, x2 + d, y2 + d], dtype=np.float32)
    return ensure_min_box_size_xyxy(resized, image_shape_hw, min_size_px=min_size_px)


def recenter_box_xyxy_at_point(
    box_xyxy: np.ndarray,
    center_xy: Sequence[float],
    image_shape_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Preserve the original box size and move its center to center_xy when possible.

    If center_xy is close to an image border, the box is shifted back inside the
    image while preserving size as much as possible. This avoids invalid prompts.
    """
    H, W = image_shape_hw
    original = np.asarray(box_xyxy, dtype=np.float32)
    original_shape = original.shape
    x1, y1, x2, y2 = original.reshape(4).tolist()

    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    cx, cy = float(center_xy[0]), float(center_xy[1])

    new_x1 = cx - box_w / 2.0
    new_x2 = cx + box_w / 2.0
    new_y1 = cy - box_h / 2.0
    new_y2 = cy + box_h / 2.0

    # Keep the box inside image bounds while preserving width/height when possible.
    if box_w >= W - 1:
        new_x1, new_x2 = 0.0, float(W - 1)
    else:
        if new_x1 < 0:
            new_x2 -= new_x1
            new_x1 = 0.0
        if new_x2 > W - 1:
            new_x1 -= new_x2 - (W - 1)
            new_x2 = float(W - 1)
        new_x1 = max(0.0, new_x1)

    if box_h >= H - 1:
        new_y1, new_y2 = 0.0, float(H - 1)
    else:
        if new_y1 < 0:
            new_y2 -= new_y1
            new_y1 = 0.0
        if new_y2 > H - 1:
            new_y1 -= new_y2 - (H - 1)
            new_y2 = float(H - 1)
        new_y1 = max(0.0, new_y1)

    recentered = np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)
    recentered = clip_box_xyxy(recentered, image_shape_hw)
    return recentered.reshape(original_shape)


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------


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


def build_dataset_index(args: argparse.Namespace) -> DatasetIndex:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    volumes_dir = Path(args.volumes_dir).expanduser().resolve() if args.volumes_dir else dataset_dir / "CT_volumes"
    masks_dir = Path(args.masks_dir).expanduser().resolve() if args.masks_dir else dataset_dir / "masks_nodules" / "nifti_data"
    annotations_csv = Path(args.annotations_csv).expanduser().resolve() if args.annotations_csv else dataset_dir / "annotations.csv"
    links_csv = Path(args.links_csv).expanduser().resolve() if args.links_csv else dataset_dir / "LUNA16_metadata_split_offical.csv"

    for p, name in [
        (dataset_dir, "dataset_dir"),
        (volumes_dir, "volumes_dir"),
        (masks_dir, "masks_dir"),
        (annotations_csv, "annotations_csv"),
        (links_csv, "links_csv"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{name} does not exist: {p}")

    df_annotations = pd.read_csv(annotations_csv)
    df_links = pd.read_csv(links_csv)
    if "seriesuid" not in df_annotations.columns:
        raise ValueError(f"{annotations_csv} must contain a 'seriesuid' column")
    if not {"SeriesID", "CID"}.issubset(df_links.columns):
        raise ValueError(f"{links_csv} must contain 'SeriesID' and 'CID' columns")

    mask_files = [
        f.name
        for f in masks_dir.iterdir()
        if f.is_file() and "mask" in f.name and "contour" in f.name and "circle" not in f.name and "nodule" in f.name
    ]
    mask_id_to_file = {}
    for fname in mask_files:
        try:
            mask_id_to_file[int(fname.split("_")[0])] = fname
        except ValueError:
            continue

    volume_ids = sorted(p.stem for p in volumes_dir.glob("*.mhd"))
    if args.only_annotated:
        annotated = set(df_annotations["seriesuid"].astype(str).tolist())
        volume_ids = [v for v in volume_ids if v in annotated]

    if args.case_list:
        wanted = [line.strip() for line in Path(args.case_list).read_text().splitlines() if line.strip()]
        wanted = [w[:-4] if w.endswith(".mhd") else w for w in wanted]
        wanted_set = set(wanted)
        volume_ids = [v for v in volume_ids if v in wanted_set]

    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        volume_ids = list(rng.permutation(volume_ids))

    if args.dataset_fraction is not None:
        if not (0 < args.dataset_fraction <= 1):
            raise ValueError("--dataset-fraction must be in (0, 1]. Use 0.10 for 10%.")
        n = max(1, int(math.ceil(len(volume_ids) * args.dataset_fraction)))
        volume_ids = volume_ids[:n]

    if args.max_cases is not None:
        volume_ids = volume_ids[: max(0, args.max_cases)]

    if args.shard_count > 1:
        if not (0 <= args.shard_index < args.shard_count):
            raise ValueError("--shard-index must be in [0, shard_count).")
        volume_ids = [v for i, v in enumerate(volume_ids) if i % args.shard_count == args.shard_index]

    return DatasetIndex(
        dataset_dir=dataset_dir,
        volumes_dir=volumes_dir,
        masks_dir=masks_dir,
        annotations_csv=annotations_csv,
        links_csv=links_csv,
        df_annotations=df_annotations,
        df_links=df_links,
        mask_id_to_file=mask_id_to_file,
        volume_ids=volume_ids,
    )


def load_case(index: DatasetIndex, series_id: str) -> Optional[Tuple[sitk.Image, np.ndarray, np.ndarray, Path]]:
    image_path = index.volumes_dir / f"{series_id}.mhd"
    if not image_path.is_file():
        print(f"Missing image: {image_path}")
        return None

    links = index.df_links[index.df_links["SeriesID"].astype(str) == str(series_id)]
    if len(links) == 0:
        print(f"No mask link found for: {series_id}")
        return None

    mask_id = int(links["CID"].iloc[0])
    mask_fname = index.mask_id_to_file.get(mask_id)
    if mask_fname is None:
        print(f"Mask id not found: CID={mask_id}, series={series_id}")
        return None

    mask_path = index.masks_dir / mask_fname
    if not mask_path.is_file():
        print(f"Missing mask: {mask_path}")
        return None

    image_itk = sitk.ReadImage(str(image_path))
    image_array = sitk.GetArrayFromImage(image_itk)
    mask_itk = sitk.ReadImage(str(mask_path))
    gt_volume = (sitk.GetArrayFromImage(mask_itk) >= 0.5).astype(np.uint8)

    if image_array.shape != gt_volume.shape:
        raise ValueError(f"Shape mismatch for {series_id}: image {image_array.shape}, mask {gt_volume.shape}")
    return image_itk, image_array, gt_volume, mask_path


def load_case_safe(index: DatasetIndex, series_id: str):
    """Load one case and return (series_id, loaded_tuple_or_none, exception_or_none)."""
    try:
        return series_id, load_case(index, series_id), None
    except Exception as exc:
        return series_id, None, exc


def iter_loaded_cases(index: DatasetIndex, prefetch_cases: int):
    """
    Ordered iterator over loaded cases.

    prefetch_cases > 0 overlaps CPU/SimpleITK I/O with GPU inference. This is the
    useful DataLoader-like optimization here, because SAM2 itself does not expose
    a clean batch-of-volumes inference API for different CT volumes.
    """
    ids = list(index.volume_ids)
    if prefetch_cases <= 0:
        for series_id in ids:
            yield load_case_safe(index, series_id)
        return

    max_workers = max(1, int(prefetch_cases))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        iterator = iter(ids)
        futures = {}

        for _ in range(min(max_workers, len(ids))):
            series_id = next(iterator)
            futures[series_id] = executor.submit(load_case_safe, index, series_id)

        for series_id in ids:
            fut = futures.pop(series_id)
            try:
                next_series_id = next(iterator)
                futures[next_series_id] = executor.submit(load_case_safe, index, next_series_id)
            except StopIteration:
                pass
            yield fut.result()


def write_pred_volume(pred: np.ndarray, reference_image: sitk.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_img = sitk.GetImageFromArray(pred.astype(np.uint8))
    pred_img.CopyInformation(reference_image)
    sitk.WriteImage(pred_img, str(out_path))


# -----------------------------------------------------------------------------
# Prompt extraction for image and video modes
# -----------------------------------------------------------------------------


def get_interior_point(component_mask: np.ndarray) -> List[float]:
    component_mask = (component_mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return [float(x), float(y)]


def extract_blobs_from_slice(
    mask_2d: np.ndarray,
    box_size_perturb_px: int = 0,
    box_min_size_px: int = 3,
) -> List[Dict]:
    mask_bin = (mask_2d > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    H, W = mask_bin.shape
    for contour in contours:
        if contour is None or len(contour) == 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        component_mask = np.zeros_like(mask_bin, dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], contourIdx=-1, color=1, thickness=-1)
        if component_mask.sum() == 0:
            continue
        blobs.append(
            {
                "center": get_interior_point(component_mask),
                "bbox": resize_box_xyxy(
                    [int(x), int(y), int(x + w - 1), int(y + h - 1)],
                    image_shape_hw=(H, W),
                    delta_px=box_size_perturb_px,
                    min_size_px=box_min_size_px,
                ).astype(np.float32).tolist(),
                "component_mask": component_mask,
                "contour": contour,
            }
        )
    # Make blob indexing deterministic for prompt perturbation identities.
    blobs.sort(key=lambda b: (b["bbox"][1], b["bbox"][0], b["bbox"][3], b["bbox"][2]))
    return blobs


def build_slice_prompt_dict(
    mask_3d: np.ndarray,
    box_size_perturb_px: int = 0,
    box_min_size_px: int = 3,
) -> List[Dict]:
    mask_3d = (mask_3d > 0).astype(np.uint8)
    slice_dicts = []
    for z in range(mask_3d.shape[0]):
        mask_slice = mask_3d[z]
        if not np.any(mask_slice > 0):
            continue
        blobs = extract_blobs_from_slice(
            mask_slice,
            box_size_perturb_px=box_size_perturb_px,
            box_min_size_px=box_min_size_px,
        )
        if not blobs:
            continue
        slice_dicts.append(
            {
                "z": z,
                "mask_slice": mask_slice,
                "blobs": blobs,
                "point_coords": np.array([b["center"] for b in blobs], dtype=np.float32),
                "point_labels": np.ones(len(blobs), dtype=np.int32),
                "boxes": np.array([b["bbox"] for b in blobs], dtype=np.float32),
            }
        )
    return slice_dicts


def get_3d_connected_components(mask_zyx: np.ndarray) -> Tuple[np.ndarray, int]:
    structure = ndimage.generate_binary_structure(rank=3, connectivity=2)
    return ndimage.label(mask_zyx.astype(np.uint8), structure=structure)


def get_component_center_point_3d(component_mask: np.ndarray) -> Tuple[int, int, int]:
    dist = ndimage.distance_transform_edt(component_mask)
    return tuple(int(v) for v in np.unravel_index(np.argmax(dist), dist.shape))


def get_bbox_from_2d_mask(mask_2d: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def extract_nodule_prompts_from_3d_mask(
    mask_zyx: np.ndarray,
    box_size_perturb_px: int = 0,
    box_min_size_px: int = 3,
) -> List[Dict]:
    labeled, num = get_3d_connected_components(mask_zyx)
    prompts = []
    obj_id = 1
    for k in range(1, num + 1):
        comp = labeled == k
        if comp.sum() == 0:
            continue
        z, y, x = get_component_center_point_3d(comp)
        box_xyxy = get_bbox_from_2d_mask(comp[z].astype(np.uint8))
        if box_xyxy is None:
            continue
        box_xyxy = resize_box_xyxy(
            box_xyxy,
            image_shape_hw=mask_zyx.shape[1:],
            delta_px=box_size_perturb_px,
            min_size_px=box_min_size_px,
        )
        prompts.append(
            {
                "obj_id": obj_id,
                "center_zyx": (int(z), int(y), int(x)),
                "frame_idx": int(z),
                "point_xy": np.array([[float(x), float(y)]], dtype=np.float32),
                "point_labels": np.array([1], dtype=np.int32),
                "box_xyxy": box_xyxy,
            }
        )
        obj_id += 1
    return prompts


def perturb_slice_prompt_dicts(
    slice_dicts: List[Dict],
    series_id: str,
    image_shape_hw: Tuple[int, int],
    rmax: int,
    seed: int,
) -> List[Dict]:
    """
    Apply deterministic xy perturbations to slice-wise point and box prompts.

    For each blob on each slice, the point is shifted by the sampled (dx, dy), and
    the corresponding box is recentered on the shifted point while preserving its
    original size as much as possible. The same perturbed prompt is then reused by
    point, box, and point+box outputs.
    """
    rmax = int(rmax)
    if rmax <= 0:
        return slice_dicts

    perturbed = []
    for info in slice_dicts:
        new_info = dict(info)
        z = int(info["z"])
        points = np.asarray(info["point_coords"], dtype=np.float32).copy()
        boxes = np.asarray(info["boxes"], dtype=np.float32).copy()
        shifts = []

        for i in range(points.shape[0]):
            dx, dy = sample_prompt_shift_xy(
                base_seed=seed,
                series_id=series_id,
                prompt_scope="image-slice",
                z=z,
                prompt_index=i,
                rmax=rmax,
            )
            shifted_point = clip_point_xy(points[i] + np.array([dx, dy], dtype=np.float32), image_shape_hw)
            points[i] = shifted_point
            boxes[i] = recenter_box_xyxy_at_point(boxes[i], shifted_point, image_shape_hw)
            shifts.append([dx, dy])

        new_info["point_coords"] = points
        new_info["boxes"] = boxes
        new_info["perturb_shifts_xy"] = np.array(shifts, dtype=np.int32)
        perturbed.append(new_info)

    return perturbed


def perturb_3d_nodule_prompts(
    prompts: List[Dict],
    series_id: str,
    image_shape_hw: Tuple[int, int],
    volume_depth: int,
    rmax: int,
    seed: int,
    z_rmax: int = 0,
    z_only: bool = False,
) -> List[Dict]:
    """
    Apply deterministic perturbations to video-mode 3D object prompts.

    By default, --prompt-perturb-rmax perturbs only x/y. The separate
    --video-z-perturb-rmax option perturbs the prompt frame by dz in
    [-z_rmax, +z_rmax], clipped to [0, Z - 1].

    If z_only=True, x/y perturbation is disabled for video mode even when
    --prompt-perturb-rmax is > 0.
    """
    rmax = int(rmax)
    z_rmax = int(z_rmax)
    if rmax <= 0 and z_rmax <= 0:
        return prompts

    perturbed = []
    for p in prompts:
        new_p = dict(p)
        z0 = int(p["frame_idx"])
        obj_id = int(p["obj_id"])

        dz = sample_prompt_shift_z(
            base_seed=seed,
            series_id=series_id,
            prompt_scope="video-object-z",
            z=z0,
            prompt_index=obj_id,
            rmax=z_rmax,
        )
        z_new = clip_z_index(z0 + dz, volume_depth)

        dx, dy = 0, 0
        shifted_point = np.asarray(p["point_xy"], dtype=np.float32).copy()
        if rmax > 0 and not z_only:
            dx, dy = sample_prompt_shift_xy(
                base_seed=seed,
                series_id=series_id,
                prompt_scope="video-object",
                z=z0,
                prompt_index=obj_id,
                rmax=rmax,
            )
            shifted_point = clip_point_xy(
                shifted_point + np.array([[dx, dy]], dtype=np.float32),
                image_shape_hw,
            )

        new_p["frame_idx"] = int(z_new)
        new_p["point_xy"] = shifted_point
        new_p["box_xyxy"] = recenter_box_xyxy_at_point(p["box_xyxy"], shifted_point[0], image_shape_hw)
        new_p["perturb_shift_xy"] = (int(dx), int(dy))
        new_p["perturb_shift_z"] = int(dz)
        new_p["center_zyx_perturbed"] = (
            int(z_new),
            int(round(float(shifted_point[0, 1]))),
            int(round(float(shifted_point[0, 0]))),
        )
        perturbed.append(new_p)

    return perturbed

# -----------------------------------------------------------------------------
# Automatic mask filtering
# -----------------------------------------------------------------------------


def compute_mask_shape_features(mask: np.ndarray) -> Optional[Dict]:
    mask = (mask > 0).astype(np.uint8)
    area = int(mask.sum())
    if area == 0:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / float(h) if h > 0 else np.inf
    elongation = max(aspect_ratio, 1.0 / max(aspect_ratio, 1e-8))
    circularity = 4.0 * math.pi * area / (perimeter**2) if perimeter > 0 else 0.0
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = float(area) / float(hull_area) if hull_area > 0 else 0.0
    return {
        "area": area,
        "perimeter": float(perimeter),
        "circularity": float(circularity),
        "solidity": float(solidity),
        "bbox": [int(x), int(y), int(x + w - 1), int(y + h - 1)],
        "width": int(w),
        "height": int(h),
        "aspect_ratio": float(aspect_ratio),
        "elongation": float(elongation),
    }


def is_nodule_like_mask(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
    min_circularity: float,
    min_solidity: float,
    max_elongation: float,
) -> Tuple[bool, Optional[Dict]]:
    feats = compute_mask_shape_features(mask)
    if feats is None:
        return False, None
    keep = (
        feats["area"] >= min_area
        and feats["area"] <= max_area
        and feats["circularity"] >= min_circularity
        and feats["solidity"] >= min_solidity
        and feats["elongation"] <= max_elongation
    )
    return keep, feats


def merge_sam2_automatic_masks_blob_like(
    anns: List[Dict],
    image_shape: Tuple[int, int],
    min_area: int,
    max_area: int,
    min_circularity: float,
    min_solidity: float,
    max_elongation: float,
    pred_iou_thresh: Optional[float],
    stability_score_thresh: Optional[float],
) -> np.ndarray:
    H, W = image_shape
    merged = np.zeros((H, W), dtype=bool)
    for ann in anns:
        seg = ann["segmentation"].astype(np.uint8)
        pred_iou = ann.get("predicted_iou")
        stability = ann.get("stability_score")
        if pred_iou_thresh is not None and pred_iou is not None and pred_iou < pred_iou_thresh:
            continue
        if stability_score_thresh is not None and stability is not None and stability < stability_score_thresh:
            continue
        keep, _ = is_nodule_like_mask(seg, min_area, max_area, min_circularity, min_solidity, max_elongation)
        if keep:
            merged |= seg.astype(bool)
    return merged.astype(np.uint8)


# -----------------------------------------------------------------------------
# SAM2 model builders
# -----------------------------------------------------------------------------


def build_image_predictor(args: argparse.Namespace, device: torch.device) -> SAM2ImagePredictor:
    sam2_model = build_sam2(args.model_cfg, args.checkpoint, device=device)
    return SAM2ImagePredictor(sam2_model)


def build_mask_generator(args: argparse.Namespace, device: torch.device) -> SAM2AutomaticMaskGenerator:
    sam2_model = build_sam2(args.model_cfg, args.checkpoint, device=device)
    return SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=args.auto_points_per_side,
        pred_iou_thresh=args.auto_generator_pred_iou_thresh,
        stability_score_thresh=args.auto_generator_stability_score_thresh,
        crop_n_layers=args.auto_crop_n_layers,
        crop_n_points_downscale_factor=args.auto_crop_n_points_downscale_factor,
        min_mask_region_area=args.auto_min_mask_region_area,
    )


def build_video_predictor(args: argparse.Namespace, device: torch.device):
    return build_sam2_video_predictor(
        args.model_cfg,
        args.checkpoint,
        device=device,
        vos_optimized=args.vos_optimized,
    )


# -----------------------------------------------------------------------------
# Image modes
# -----------------------------------------------------------------------------


def predict_image_single(
    series_id: str,
    image_array: np.ndarray,
    gt_volume: np.ndarray,
    predictor: Optional[SAM2ImagePredictor],
    mask_generator: Optional[SAM2AutomaticMaskGenerator],
    args: argparse.Namespace,
    device: torch.device,
):
    prompts = build_slice_prompt_dict(
        gt_volume,
        box_size_perturb_px=args.box_size_perturb_px,
        box_min_size_px=args.box_min_size_px,
    )
    prompts = perturb_slice_prompt_dicts(
        prompts,
        series_id=series_id,
        image_shape_hw=gt_volume.shape[1:],
        rmax=args.prompt_perturb_rmax,
        seed=args.seed,
    )
    if len(prompts) == 0:
        raise ValueError("No valid GT-positive prompt slices found")
    prompt_by_z = {d["z"]: d for d in prompts}

    requested = set(args.image_outputs)
    need_auto = "auto" in requested
    need_prompted = any(m in requested for m in PROMPT_MODES)

    if args.process_all_slices_auto:
        auto_z_list = list(range(image_array.shape[0]))
    else:
        auto_z_list = sorted(prompt_by_z.keys())

    z_to_process = set()
    if need_auto:
        z_to_process.update(auto_z_list)
    if need_prompted:
        z_to_process.update(prompt_by_z.keys())

    pred = {mode: np.zeros_like(gt_volume, dtype=bool) for mode in requested}
    scores = {"point": [], "box": [], "point+box": []}

    for z in sorted(z_to_process):
        if need_auto and z in auto_z_list:
            if mask_generator is None:
                raise RuntimeError("Automatic mode requested but mask_generator is None")
            img_auto = build_sam2_input_slice(image_array, z, use_triplet_channels=args.use_triplet_channels)
            with torch.inference_mode(), safe_autocast(device, args.amp_dtype):
                anns = mask_generator.generate(img_auto)
            pred_slice = merge_sam2_automatic_masks_blob_like(
                anns=anns,
                image_shape=img_auto.shape[:2],
                min_area=args.auto_min_area,
                max_area=args.auto_max_area,
                min_circularity=args.auto_min_circularity,
                min_solidity=args.auto_min_solidity,
                max_elongation=args.auto_max_elongation,
                pred_iou_thresh=args.auto_filter_pred_iou_thresh,
                stability_score_thresh=args.auto_filter_stability_score_thresh,
            )
            pred["auto"][z] |= pred_slice.astype(bool)

        if need_prompted and z in prompt_by_z:
            if predictor is None:
                raise RuntimeError("Prompted image mode requested but predictor is None")
            info = prompt_by_z[z]
            img = build_sam2_input_slice(image_array, z, use_triplet_channels=args.use_triplet_channels)
            with torch.inference_mode(), safe_autocast(device, args.amp_dtype):
                predictor.set_image(img)
                slice_acc = {m: np.zeros_like(gt_volume[z], dtype=bool) for m in requested if m in PROMPT_MODES}
                for i in range(len(info["blobs"])):
                    point = info["point_coords"][i : i + 1]
                    label = info["point_labels"][i : i + 1]
                    box = info["boxes"][i : i + 1]
                    if "point" in requested:
                        masks, sc, _ = predictor.predict(point_coords=point, point_labels=label, multimask_output=False)
                        slice_acc["point"] |= masks[0].astype(bool)
                        scores["point"].append(float(sc[0]))
                    if "box" in requested:
                        masks, sc, _ = predictor.predict(point_coords=None, point_labels=None, box=box, multimask_output=False)
                        slice_acc["box"] |= masks[0].astype(bool)
                        scores["box"].append(float(sc[0]))
                    if "point+box" in requested:
                        masks, sc, _ = predictor.predict(point_coords=point, point_labels=label, box=box, multimask_output=False)
                        slice_acc["point+box"] |= masks[0].astype(bool)
                        scores["point+box"].append(float(sc[0]))
                for mode, sl in slice_acc.items():
                    pred[mode][z] |= sl

    pred = {k: v.astype(np.uint8) for k, v in pred.items()}
    row = {"VolumeID": series_id}
    for mode in args.image_outputs:
        row[f"DSC ({mode})"] = dice_score(gt_volume, pred[mode])
    for mode in PROMPT_MODES:
        if scores[mode]:
            row[f"score_mean ({mode})"] = float(np.mean(scores[mode]))
            row[f"score_n ({mode})"] = len(scores[mode])
    return row, pred


def predict_image_multi(
    series_id: str,
    image_array: np.ndarray,
    gt_volume: np.ndarray,
    predictor: SAM2ImagePredictor,
    args: argparse.Namespace,
    device: torch.device,
):
    prompts = build_slice_prompt_dict(
        gt_volume,
        box_size_perturb_px=args.box_size_perturb_px,
        box_min_size_px=args.box_min_size_px,
    )
    prompts = perturb_slice_prompt_dicts(
        prompts,
        series_id=series_id,
        image_shape_hw=gt_volume.shape[1:],
        rmax=args.prompt_perturb_rmax,
        seed=args.seed,
    )
    if len(prompts) == 0:
        raise ValueError("No valid GT-positive prompt slices found")

    requested = [m for m in args.image_outputs if m in PROMPT_MODES]
    pred = {f"{mode}_ch{ch}": np.zeros_like(gt_volume, dtype=bool) for mode in requested for ch in range(3)}
    scores = {f"{mode}_ch{ch}": [] for mode in requested for ch in range(3)}

    for info in prompts:
        z = info["z"]
        img = build_sam2_input_slice(image_array, z, use_triplet_channels=args.use_triplet_channels)
        with torch.inference_mode(), safe_autocast(device, args.amp_dtype):
            predictor.set_image(img)
            slice_acc = {k: np.zeros_like(gt_volume[z], dtype=bool) for k in pred}
            for i in range(len(info["blobs"])):
                point = info["point_coords"][i : i + 1]
                label = info["point_labels"][i : i + 1]
                box = info["boxes"][i : i + 1]
                outputs = {}
                if "point" in requested:
                    outputs["point"] = predictor.predict(point_coords=point, point_labels=label, multimask_output=True)[:2]
                if "box" in requested:
                    outputs["box"] = predictor.predict(point_coords=None, point_labels=None, box=box, multimask_output=True)[:2]
                if "point+box" in requested:
                    outputs["point+box"] = predictor.predict(point_coords=point, point_labels=label, box=box, multimask_output=True)[:2]
                for mode, (masks, sc) in outputs.items():
                    n_masks = min(3, masks.shape[0])
                    for ch in range(n_masks):
                        key = f"{mode}_ch{ch}"
                        slice_acc[key] |= masks[ch].astype(bool)
                        scores[key].append(float(sc[ch]))
            for key, sl in slice_acc.items():
                pred[key][z] |= sl

    pred = {k: v.astype(np.uint8) for k, v in pred.items()}
    row = {"VolumeID": series_id}
    for key, vol in pred.items():
        row[f"DSC ({key})"] = dice_score(gt_volume, vol)
        if scores[key]:
            row[f"score_mean ({key})"] = float(np.mean(scores[key]))
            row[f"score_n ({key})"] = len(scores[key])
    return row, pred


# -----------------------------------------------------------------------------
# Video mode
# -----------------------------------------------------------------------------


def write_volume_as_sam2_frames(volume_zyx: np.ndarray, out_dir: Path, use_triplet_channels: bool, frame_ext: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for z in range(volume_zyx.shape[0]):
        rgb = build_sam2_input_slice(volume_zyx, z, use_triplet_channels=use_triplet_channels)
        Image.fromarray(rgb).save(out_dir / f"{z:05d}{frame_ext}")


def add_sam2_prompt_for_object(
    predictor,
    inference_state,
    frame_idx: int,
    obj_id: int,
    prompt_mode: str,
    point_xy: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
    box_xyxy: Optional[np.ndarray],
):
    if prompt_mode == "point":
        return predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=point_xy,
            labels=point_labels,
        )
    if prompt_mode == "box":
        return predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box_xyxy,
        )
    if prompt_mode == "point+box":
        return predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=point_xy,
            labels=point_labels,
            box=box_xyxy,
        )
    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


def accumulate_video_propagation(predictor, inference_state, pred_by_obj: Dict[int, np.ndarray], reverse: bool):
    """Propagate and OR into pred_by_obj. reverse=True is important for full-volume masks."""
    kwargs = {"reverse": True} if reverse else {}
    try:
        iterator = predictor.propagate_in_video(inference_state, **kwargs)
    except TypeError:
        if reverse:
            print("WARNING: This SAM2 predictor does not support reverse=True; slices before prompt may be missed.")
            return
        iterator = predictor.propagate_in_video(inference_state)

    for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
        for i, obj_id in enumerate(out_obj_ids):
            mask_np = out_mask_logits[i].squeeze().float().cpu().numpy()
            mask_bin = (mask_np > 0.0).astype(np.uint8)
            pred_by_obj[int(obj_id)][int(out_frame_idx)] |= mask_bin


def predict_video(
    series_id: str,
    image_array: np.ndarray,
    gt_volume: np.ndarray,
    predictor,
    args: argparse.Namespace,
    device: torch.device,
):
    prompts = extract_nodule_prompts_from_3d_mask(
        gt_volume,
        box_size_perturb_px=args.box_size_perturb_px,
        box_min_size_px=args.box_min_size_px,
    )
    prompts = perturb_3d_nodule_prompts(
        prompts,
        series_id=series_id,
        image_shape_hw=gt_volume.shape[1:],
        volume_depth=gt_volume.shape[0],
        rmax=args.prompt_perturb_rmax,
        seed=args.seed,
        z_rmax=args.video_z_perturb_rmax,
        z_only=args.video_z_perturb_only,
    )
    if len(prompts) == 0:
        raise ValueError("No 3D connected nodules found in mask")

    temp_parent = Path(args.frame_tmp_dir or os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir())
    temp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_{series_id}_", dir=str(temp_parent)))

    try:
        write_volume_as_sam2_frames(
            image_array,
            temp_dir,
            use_triplet_channels=args.use_triplet_channels,
            frame_ext=args.frame_ext,
        )
        row = {"VolumeID": series_id, "n_prompts": len(prompts)}
        volumes = {}
        Z, H, W = gt_volume.shape

        for prompt_mode in args.prompt_modes:
            with torch.inference_mode(), safe_autocast(device, args.amp_dtype):
                inference_state = predictor.init_state(video_path=str(temp_dir))
                if hasattr(predictor, "reset_state"):
                    predictor.reset_state(inference_state)

                for p in prompts:
                    add_sam2_prompt_for_object(
                        predictor=predictor,
                        inference_state=inference_state,
                        frame_idx=p["frame_idx"],
                        obj_id=p["obj_id"],
                        prompt_mode=prompt_mode,
                        point_xy=p["point_xy"],
                        point_labels=p["point_labels"],
                        box_xyxy=p["box_xyxy"],
                    )

                pred_by_obj = {p["obj_id"]: np.zeros((Z, H, W), dtype=np.uint8) for p in prompts}
                accumulate_video_propagation(predictor, inference_state, pred_by_obj, reverse=False)
                if args.video_bidirectional:
                    accumulate_video_propagation(predictor, inference_state, pred_by_obj, reverse=True)

            pred_volume = np.zeros_like(gt_volume, dtype=np.uint8)
            for obj_mask in pred_by_obj.values():
                pred_volume |= obj_mask.astype(np.uint8)
            row[f"DSC (video {prompt_mode})"] = dice_score(gt_volume, pred_volume)
            volumes[f"video_{prompt_mode.replace('+', '_')}"] = pred_volume

        return row, volumes
    finally:
        if args.cleanup_frames:
            shutil.rmtree(temp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# CLI + main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified SAM2 LUNA CT inference/evaluation CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument("--dataset-dir", required=True, help="Root LUNA dataset directory.")
    parser.add_argument("--volumes-dir", default=None, help="Directory containing .mhd CT volumes. Defaults to DATASET_DIR/CT_volumes.")
    parser.add_argument("--masks-dir", default=None, help="Directory containing nodule mask files. Defaults to DATASET_DIR/masks_nodules/nifti_data.")
    parser.add_argument("--annotations-csv", default=None, help="annotations.csv path. Defaults to DATASET_DIR/annotations.csv.")
    parser.add_argument("--links-csv", default=None, help="SeriesID/CID metadata CSV. Defaults to DATASET_DIR/LUNA16_metadata_split_offical.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument("--run-name", default="sam2_predictions", help="Prefix for CSV and volume output folder.")

    # SAM2 model
    parser.add_argument("--model-cfg", required=True, help="SAM2 model config, e.g. configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--checkpoint", required=True, help="SAM2 checkpoint path, e.g. sam2.1_hiera_large.pt")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu, or mps.")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "none"], default="bf16", help="CUDA autocast dtype. Use bf16 on H200.")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True, help="Enable TF32 for CUDA matmul/cuDNN.")
    parser.add_argument("--vos-optimized", action="store_true", help="Use SAM2 video VOS optimized mode if available.")

    # Mode selection
    parser.add_argument("--mode", choices=["video", "image-single", "image-multi", "auto"], required=True)
    parser.add_argument(
        "--prompt-modes",
        nargs="+",
        choices=PROMPT_MODES,
        default=["point", "box", "point+box"],
        help="Prompt modes for video mode.",
    )
    parser.add_argument(
        "--image-outputs",
        nargs="+",
        choices=IMAGE_MODES,
        default=["auto", "point", "box", "point+box"],
        help="Outputs for image-single/auto. For image-multi, auto is ignored.",
    )
    parser.add_argument("--use-triplet-channels", action="store_true", help="Use z-1/z/z+1 as RGB channels instead of duplicated slice.")
    parser.add_argument(
        "--prompt-perturb-rmax",
        type=int,
        default=0,
        help=(
            "Maximum absolute random perturbation in pixels for prompted point/box inputs. "
            "For each prompt, dx and dy are sampled deterministically from [-rmax, +rmax]. "
            "The shifted point and recentered box are shared by point, box, and point+box modes. "
            "Use 0 to disable perturbations."
        ),
    )
    parser.add_argument(
        "--box-size-perturb-px",
        type=int,
        default=0,
        help=(
            "Resize prompt boxes by this many pixels in every direction before inference. "
            "0 keeps the original mask-derived box, positive values enlarge it, and "
            "negative values shrink it."
        ),
    )
    parser.add_argument(
        "--box-min-size-px",
        type=int,
        default=3,
        help=(
            "Minimum inclusive side length for prompt boxes after box-size perturbation. "
            "This prevents negative or too-small boxes when using negative perturbations."
        ),
    )
    parser.add_argument(
        "--video-z-perturb-rmax",
        type=int,
        default=0,
        help=(
            "Video-mode only: maximum absolute deterministic random perturbation in slices "
            "for the prompt frame. dz is sampled from [-rmax, +rmax] and clipped to "
            "[0, Z - 1]. Use 0 to disable z perturbation."
        ),
    )
    parser.add_argument(
        "--video-z-perturb-only",
        action="store_true",
        help=(
            "Video-mode only: perturb only the z/frame coordinate. When enabled, "
            "--prompt-perturb-rmax is ignored for x/y in video mode, but still applies "
            "to image-single/image-multi modes."
        ),
    )

    # Automatic mask options
    parser.add_argument("--process-all-slices-auto", action="store_true", help="Run automatic mask generator on all slices, not only GT-positive slices.")
    parser.add_argument("--auto-min-area", type=int, default=5)
    parser.add_argument("--auto-max-area", type=int, default=400)
    parser.add_argument("--auto-min-circularity", type=float, default=0.20)
    parser.add_argument("--auto-min-solidity", type=float, default=0.70)
    parser.add_argument("--auto-max-elongation", type=float, default=2.0)
    parser.add_argument("--auto-filter-pred-iou-thresh", type=float, default=None, help="Post-filter generated masks by predicted_iou.")
    parser.add_argument("--auto-filter-stability-score-thresh", type=float, default=None, help="Post-filter generated masks by stability_score.")
    parser.add_argument("--auto-points-per-side", type=int, default=32, help="Lower this to reduce automatic mask generator cost.")
    parser.add_argument("--auto-generator-pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--auto-generator-stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--auto-crop-n-layers", type=int, default=0, help="0 is cheaper; >0 can improve small-object recall but costs more.")
    parser.add_argument("--auto-crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--auto-min-mask-region-area", type=int, default=0)

    # Video options
    parser.add_argument("--video-bidirectional", action=argparse.BooleanOptionalAction, default=True, help="Propagate both forward and reverse from prompt slice. Keep true for full-volume segmentation.")
    parser.add_argument("--frame-tmp-dir", default=None, help="Temporary frame directory parent. Defaults to SLURM_TMPDIR or system tmp.")
    parser.add_argument("--frame-ext", choices=[".jpg", ".png"], default=".jpg", help="Frame image format for SAM2 video predictor. PNG is lossless but larger.")
    parser.add_argument("--cleanup-frames", action=argparse.BooleanOptionalAction, default=True)

    # Dataset selection / cluster efficiency
    parser.add_argument("--only-annotated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-fraction", type=float, default=None, help="Analyze a fraction of selected dataset, e.g. 0.10 for 10%.")
    parser.add_argument("--max-cases", type=int, default=None, help="Cap number of volumes after filtering/subsampling.")
    parser.add_argument("--case-list", default=None, help="Text file with one SeriesID per line.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before fraction/max/sharding selection.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--shard-index",
        nargs="?",
        type=int,
        const=env_int("SLURM_ARRAY_TASK_ID", 0),
        default=env_int("SLURM_ARRAY_TASK_ID", 0),
        help=(
            "Shard index. If the flag is provided without a value, the script uses "
            "SLURM_ARRAY_TASK_ID or 0. This avoids argparse failures when the "
            "SLURM variable is empty."
        ),
    )
    parser.add_argument(
        "--shard-count",
        nargs="?",
        type=int,
        const=env_int("SLURM_ARRAY_TASK_COUNT", 1),
        default=env_int("SLURM_ARRAY_TASK_COUNT", 1),
        help=(
            "Number of shards. If omitted after the flag, uses SLURM_ARRAY_TASK_COUNT or 1. "
            "For SLURM arrays you can also set this manually to the total number of jobs."
        ),
    )
    parser.add_argument("--prefetch-cases", type=int, default=0, help="CPU/SimpleITK case prefetch workers. Try 1-2 on a cluster; higher uses more RAM.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip cases whose requested prediction volumes already exist. Useful for resuming.")
    parser.add_argument("--empty-cache-every", type=int, default=10, help="Call torch.cuda.empty_cache every N cases; 0 disables.")

    # Saving
    parser.add_argument("--save-volumes", action="store_true", help="Also save predicted volumes. If false, only CSV is saved.")
    parser.add_argument("--save-gt-volume", action="store_true", help="Save GT mask next to predictions for inspection.")
    parser.add_argument("--csv-filename", default=None, help="Override CSV file name.")
    parser.add_argument("--fail-fast", action="store_true", help="Raise errors immediately instead of recording them in CSV.")
    
    parser.add_argument("--create-experiment-dir", action=argparse.BooleanOptionalAction, default=True,
    help=("Create a timestamped experiment folder inside --output-dir. "
    "Use --no-create-experiment-dir to write directly into --output-dir."),)

    parser.add_argument("--experiment-name", default=None,
            help=(
            "Optional custom experiment folder name. If not provided, a name is built "
            "from date/time and main hyperparameters."),)

    parser.add_argument("--overwrite-experiment", action="store_true", help=(
            "Allow writing into an existing experiment folder. Useful only when using "
            "a fixed --experiment-name."),)

    args = parser.parse_args()
    if args.prompt_perturb_rmax < 0:
        raise ValueError("--prompt-perturb-rmax must be >= 0")
    if args.box_min_size_px < 1:
        raise ValueError("--box-min-size-px must be >= 1")
    if args.video_z_perturb_rmax < 0:
        raise ValueError("--video-z-perturb-rmax must be >= 0")
    if args.mode == "auto":
        args.image_outputs = ["auto"]
    if args.mode == "image-multi":
        args.image_outputs = [m for m in args.image_outputs if m in PROMPT_MODES]
        if not args.image_outputs:
            raise ValueError("image-multi requires at least one prompted output: point, box, or point+box")
    return args


def expected_volume_paths(out_dir: Path, run_name: str, series_id: str, keys: Sequence[str]) -> List[Path]:
    return [
        out_dir / "predicted_volumes" / key / f"{series_id}_{key}.nii.gz"
        for key in keys
    ]


def main():

    args = parse_args()

    device = select_device(args.device)
    configure_torch(device, args.amp_dtype, args.allow_tf32)

    out_dir = setup_experiment_dir(args)

    print(f"Using device: {device}; mode={args.mode}; amp={args.amp_dtype}; tf32={args.allow_tf32}")
    print(f"Experiment directory: {out_dir}")

    index = build_dataset_index(args)

    save_experiment_config(args=args, exp_dir=out_dir, device=device, index=index)
    print(f"Saved config: {out_dir / 'config.yaml'}")
    print(f"Selected {len(index.volume_ids)} volumes for this run/shard.")
    print(f"Shard index/count: {args.shard_index}/{args.shard_count}; prefetch_cases={args.prefetch_cases}")

    # Build only the models required by the selected mode/output to avoid wasting H200 memory.
    image_predictor = None
    mask_generator = None
    video_predictor = None
    if args.mode == "video":
        video_predictor = build_video_predictor(args, device)
    elif args.mode == "image-single":
        if any(m in args.image_outputs for m in PROMPT_MODES):
            image_predictor = build_image_predictor(args, device)
        if "auto" in args.image_outputs:
            mask_generator = build_mask_generator(args, device)
    elif args.mode == "image-multi":
        image_predictor = build_image_predictor(args, device)
    elif args.mode == "auto":
        mask_generator = build_mask_generator(args, device)

    rows: List[Dict] = []

    csv_name = args.csv_filename or "predictions.csv"
    csv_path = out_dir / csv_name

    case_iter = iter_loaded_cases(index, args.prefetch_cases)
    for case_idx, (series_id, loaded, load_error) in enumerate(
        tqdm.tqdm(case_iter, total=len(index.volume_ids), desc="Volumes"),
        start=1,
    ):
        try:
            if load_error is not None:
                raise load_error
            if loaded is None:
                rows.append({"VolumeID": series_id, "status": "missing_input"})
                continue
            image_itk, image_array, gt_volume, mask_path = loaded

            if args.mode == "video":
                volume_keys = [f"video_{m.replace('+', '_')}" for m in args.prompt_modes]
            elif args.mode == "image-single" or args.mode == "auto":
                volume_keys = [m.replace("+", "_") for m in args.image_outputs]
            else:
                volume_keys = [f"{m.replace('+', '_')}_ch{ch}" for m in args.image_outputs for ch in range(3)]

            if args.skip_existing and args.save_volumes:
                paths = expected_volume_paths(out_dir, args.run_name, series_id, volume_keys)
                if all(p.exists() for p in paths):
                    rows.append({"VolumeID": series_id, "status": "skipped_existing"})
                    continue

            if args.mode == "video":
                row, pred_volumes = predict_video(series_id, image_array, gt_volume, video_predictor, args, device)
            elif args.mode == "image-single" or args.mode == "auto":
                row, pred_volumes = predict_image_single(series_id, image_array, gt_volume, image_predictor, mask_generator, args, device)
            elif args.mode == "image-multi":
                row, pred_volumes = predict_image_multi(series_id, image_array, gt_volume, image_predictor, args, device)
            else:
                raise ValueError(args.mode)

            row.update({
                "status": "ok",
                "mask_file": Path(mask_path).name,
                "n_slices": int(gt_volume.shape[0]),
                "prompt_perturb_rmax": int(args.prompt_perturb_rmax),
                "prompt_perturb_seed": int(args.seed),
                "box_size_perturb_px": int(args.box_size_perturb_px),
                "box_min_size_px": int(args.box_min_size_px),
                "video_z_perturb_rmax": int(args.video_z_perturb_rmax),
                "video_z_perturb_only": bool(args.video_z_perturb_only),
            })

            rows.append(row)

            if args.save_volumes:
                pred_root = out_dir / "predicted_volumes"

                for key, vol in pred_volumes.items():
                    safe_key = key.replace("+", "_")
                    out_path = pred_root / safe_key / f"{series_id}_{safe_key}.nii.gz"
                    write_pred_volume(vol, image_itk, out_path)

                if args.save_gt_volume:
                    out_path = pred_root / "gt" / f"{series_id}_gt.nii.gz"
                    write_pred_volume(gt_volume, image_itk, out_path)

        except Exception as exc:
            if args.fail_fast:
                raise
            rows.append({"VolumeID": series_id, "status": "error", "error": repr(exc)})
            print(f"ERROR in {series_id}: {repr(exc)}")

        if args.empty_cache_every and case_idx % args.empty_cache_every == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Incremental CSV makes long cluster jobs resumable/debuggable.
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    dice_cols = [c for c in df.columns if c.startswith("DSC")]
    for c in dice_cols:
        print(f"Mean {c}: {pd.to_numeric(df[c], errors='coerce').mean():.6f}")


if __name__ == "__main__":
    main()