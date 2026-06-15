#!/usr/bin/env python3
"""Fine-tune SAM2 for CT nodule segmentation.

This script is converted from the original notebook and adds:
  - argparse CLI
  - date/hyperparameter-based results folder creation
  - config.yaml with all initial parameters
  - CSV/JSON training history
  - latest checkpoint every epoch
  - improvement checkpoints every time validation Dice improves
  - stable best checkpoint/weights aliases
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sam2.build_sam import build_sam2


# -----------------------------
# Reproducibility and utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def safe_name(value: Any) -> str:
    return str(value).replace("/", "-").replace(" ", "")


def make_run_dir(args: argparse.Namespace) -> Path:
    base_dir = Path(args.results_root)
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.run_name:
        run_name = args.run_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [
            stamp,
            f"sam2_{Path(args.sam2_checkpoint).stem}",
            f"img{args.image_size}",
            f"T{args.num_frames}",
            f"bs{args.batch_size}",
            f"lr{args.lr:g}",
            f"wd{args.weight_decay:g}",
            f"2p5d{int(args.use_2p5d)}",
        ]
        run_name = "__".join(safe_name(p) for p in parts)

    run_dir = base_dir / run_name
    if run_dir.exists() and not args.overwrite:
        suffix = datetime.now().strftime("%H%M%S")
        run_dir = base_dir / f"{run_name}__{suffix}"
    run_dir.mkdir(parents=True, exist_ok=args.overwrite)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    return run_dir


def save_config(args: argparse.Namespace, run_dir: Path) -> None:
    config = vars(args).copy()
    config["run_dir"] = str(run_dir)
    config["created_at"] = datetime.now().isoformat(timespec="seconds")
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


# -----------------------------
# IO and preprocessing
# -----------------------------

def window_ct(x: np.ndarray, wl: float = -600, ww: float = 1500) -> np.ndarray:
    lo = wl - ww / 2
    hi = wl + ww / 2
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return x


def resize_2d(img: np.ndarray, out_size: int = 1024, order: int = 1) -> np.ndarray:
    return resize(
        img,
        (out_size, out_size),
        order=order,
        preserve_range=True,
        anti_aliasing=(order != 0),
    ).astype(np.float32)


def load_mhd(path: Path | str) -> np.ndarray:
    """Returns volume as [Z, H, W]. SimpleITK reads MHD as [Z, Y, X]."""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.float32)


def load_nii_mask(path: Path | str) -> np.ndarray:
    """Nibabel usually returns [X, Y, Z], so convert to [Z, Y, X]."""
    mask = nib.load(str(path)).get_fdata().astype(np.float32)
    return np.transpose(mask, (2, 1, 0))


def make_rgb_slice(volume: np.ndarray, z: int, use_2p5d: bool = True) -> np.ndarray:
    if use_2p5d:
        z0 = max(0, z - 1)
        z1 = z
        z2 = min(volume.shape[0] - 1, z + 1)
        return np.stack([volume[z0], volume[z1], volume[z2]], axis=0)
    s = volume[z]
    return np.stack([s, s, s], axis=0)


# -----------------------------
# Dataset
# -----------------------------

class NoduleVideoDataset(Dataset):
    def __init__(
        self,
        dict_pairs: Dict[str, Dict[str, str]],
        image_size: int = 512,
        num_frames: int = 4,
        use_2p5d: bool = True,
        positive_prob: float = 0.7,
        min_fg_pixels: int = 5,
        window_level: float = -600,
        window_width: float = 1500,
        random_window: bool = True,
        allow_negative_volumes: bool = True,
    ):
        self.dict_pairs = dict_pairs
        self.case_ids = sorted(list(dict_pairs.keys()))
        self.image_size = image_size
        self.num_frames = num_frames
        self.use_2p5d = use_2p5d
        self.positive_prob = positive_prob
        self.min_fg_pixels = min_fg_pixels
        self.window_level = window_level
        self.window_width = window_width
        self.random_window = random_window
        self.allow_negative_volumes = allow_negative_volumes
        self.valid_cases = []

        for case_id in tqdm(self.case_ids, desc="Indexing volumes"):
            path_image = Path(dict_pairs[case_id]["path_image"])
            path_mask = Path(dict_pairs[case_id]["path_mask"])
            if not path_image.is_file():
                print("Missing image:", path_image)
                continue
            if not path_mask.is_file():
                print("Missing mask:", path_mask)
                continue

            mask = load_nii_mask(path_mask)
            fg_per_slice = np.array([(mask[z] > 0).sum() for z in range(mask.shape[0])])
            positive_slices = np.where(fg_per_slice >= min_fg_pixels)[0]
            negative_slices = np.where(fg_per_slice < min_fg_pixels)[0]

            if len(positive_slices) == 0 and not allow_negative_volumes:
                print("Skipping volume without nodules:", case_id)
                continue

            self.valid_cases.append(
                {
                    "case_id": case_id,
                    "positive_slices": positive_slices,
                    "negative_slices": negative_slices,
                    "num_slices": mask.shape[0],
                    "has_positive": len(positive_slices) > 0,
                }
            )

        print(f"Valid volumes: {len(self.valid_cases)}")

    def __len__(self) -> int:
        return len(self.valid_cases)

    def _window_from_center(self, center: int, num_slices: int) -> list[int]:
        t = self.num_frames
        if num_slices <= t:
            frame_ids = list(range(num_slices))
            while len(frame_ids) < t:
                frame_ids.append(frame_ids[-1])
            return frame_ids
        start = int(center) - t // 2
        start = max(0, min(start, num_slices - t))
        return list(range(start, start + t))

    def _sample_window(self, item: Dict[str, Any]) -> Tuple[list[int], str]:
        num_slices = item["num_slices"]
        positive_slices = item["positive_slices"]
        negative_slices = item["negative_slices"]
        t = self.num_frames

        if num_slices <= t:
            frame_ids = list(range(num_slices))
            while len(frame_ids) < t:
                frame_ids.append(frame_ids[-1])
            return frame_ids, "short_padded"

        use_positive = len(positive_slices) > 0 and random.random() < self.positive_prob
        if use_positive:
            return self._window_from_center(int(random.choice(positive_slices)), num_slices), "positive"

        if len(negative_slices) > 0:
            return self._window_from_center(int(random.choice(negative_slices)), num_slices), "negative"

        start = random.randint(0, num_slices - t) if self.random_window else (num_slices - t) // 2
        return list(range(start, start + t)), "fallback"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.valid_cases[idx]
        case_id = item["case_id"]
        path_image = Path(self.dict_pairs[case_id]["path_image"])
        path_mask = Path(self.dict_pairs[case_id]["path_mask"])

        vol = load_mhd(path_image)
        mask = load_nii_mask(path_mask)
        if vol.shape != mask.shape:
            raise ValueError(f"Shape mismatch for {case_id}: image {vol.shape}, mask {mask.shape}")

        vol = window_ct(vol, wl=self.window_level, ww=self.window_width)
        frame_ids, sample_type = self._sample_window(item)

        video_frames, video_masks = [], []
        for z in frame_ids:
            img = make_rgb_slice(vol, z, use_2p5d=self.use_2p5d)
            msk = (mask[z] > 0).astype(np.float32)
            img = np.stack([resize_2d(img[c], self.image_size, order=1) for c in range(3)], axis=0)
            msk = resize_2d(msk, self.image_size, order=0)
            msk = (msk > 0.5).astype(np.float32)
            video_frames.append(img)
            video_masks.append(msk[None])

        return {
            "image": torch.from_numpy(np.stack(video_frames, axis=0)).float(),
            "mask": torch.from_numpy(np.stack(video_masks, axis=0)).float(),
            "case_id": case_id,
            "frame_ids": torch.tensor(frame_ids).long(),
            "sample_type": sample_type,
            "has_positive": item["has_positive"],
            "path_image": str(path_image),
            "path_mask": str(path_mask),
        }


# -----------------------------
# Model
# -----------------------------

class PromptFreeSAM2Clip(nn.Module):
    def __init__(self, sam2_model: nn.Module):
        super().__init__()
        self.sam2 = sam2_model

    def forward_one_frame(self, x: torch.Tensor) -> torch.Tensor:
        backbone_out = self.sam2.forward_image(x)
        _, vision_feats, _, feat_sizes = self.sam2._prepare_backbone_features(backbone_out)
        backbone_features = vision_feats[-1]

        b = x.shape[0]
        c = backbone_features.shape[-1]
        h, w = feat_sizes[-1]
        backbone_features = backbone_features.permute(1, 2, 0).reshape(b, c, h, w)

        high_res_features = [
            feat.permute(1, 2, 0).reshape(b, feat.shape[-1], *feat_size)
            for feat, feat_size in zip(vision_feats[:-1], feat_sizes[:-1])
        ][:2]

        outputs = self.sam2._forward_sam_heads(
            backbone_features=backbone_features,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=high_res_features,
            multimask_output=False,
        )
        return outputs[4]  # high_res_masks

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        preds = [self.forward_one_frame(video[:, t]) for t in range(video.shape[1])]
        return torch.stack(preds, dim=1)


def build_model(args: argparse.Namespace, device: str) -> nn.Module:
    if not Path(args.model_cfg).is_file():
        raise FileNotFoundError(f"model_cfg not found: {args.model_cfg}")
    if not Path(args.sam2_checkpoint).is_file():
        raise FileNotFoundError(f"sam2_checkpoint not found: {args.sam2_checkpoint}")

    sam2_base = build_sam2(args.model_cfg, args.sam2_checkpoint, device=device)
    model = PromptFreeSAM2Clip(sam2_base).to(device)

    for p in model.parameters():
        p.requires_grad = False

    if args.train_prompt_encoder:
        for p in model.sam2.sam_prompt_encoder.parameters():
            p.requires_grad = True
    if args.train_mask_decoder:
        for p in model.sam2.sam_mask_decoder.parameters():
            p.requires_grad = True
    if args.train_image_encoder:
        for p in model.sam2.image_encoder.parameters():
            p.requires_grad = True

    return model


# -----------------------------
# Splits and metrics
# -----------------------------

def create_image_mask_pairs(path_volumes: str, path_masks: str, path_ids_link_file: str) -> Dict[str, Dict[str, str]]:
    df_ids_link = pd.read_csv(path_ids_link_file)
    mhd_files = [f for f in os.listdir(path_volumes) if f.endswith(".mhd")]
    only_name_files = [f.replace(".mhd", "") for f in mhd_files]
    mask_files = [
        f for f in os.listdir(path_masks)
        if "mask" in f and "contour" in f and "circle" not in f and "nodule" in f
    ]
    mask_ids = [int(s.split("_")[0]) for s in mask_files]
    print(len(mask_files), "Masks in", path_masks)
    print(len(only_name_files), "CT volumes in", path_volumes)

    output = {}
    for mhd_file in tqdm(only_name_files, desc="Pairing images/masks"):
        sub_df = df_ids_link[df_ids_link["SeriesID"] == mhd_file]
        if len(sub_df) == 0:
            print("No mask link found for:", mhd_file)
            continue
        mask_id = sub_df["CID"].tolist()[0]
        if mask_id not in mask_ids:
            print("Mask id not found:", mask_id, mhd_file)
            continue
        idx = mask_ids.index(mask_id)
        output[mhd_file] = {
            "path_image": str(Path(path_volumes) / f"{mhd_file}.mhd"),
            "path_mask": str(Path(path_masks) / mask_files[idx]),
        }
    return output


def split_dict_pairs(
    dict_pairs: Dict[str, Dict[str, str]],
    train_size: float = 0.70,
    val_size: float = 0.20,
    test_size: float = 0.10,
    seed: int = 42,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6
    keys = sorted(list(dict_pairs.keys()))
    train_keys, temp_keys = train_test_split(keys, train_size=train_size, random_state=seed, shuffle=True)
    relative_val_size = val_size / (val_size + test_size)
    val_keys, test_keys = train_test_split(temp_keys, train_size=relative_val_size, random_state=seed, shuffle=True)
    return ({k: dict_pairs[k] for k in train_keys}, {k: dict_pairs[k] for k in val_keys}, {k: dict_pairs[k] for k in test_keys})


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    num = 2 * (probs * targets).sum(dim=dims)
    den = probs.sum(dim=dims) + targets.sum(dim=dims)
    return 1 - ((num + eps) / (den + eps)).mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    dims = tuple(range(1, preds.ndim))
    num = 2 * (preds * targets).sum(dim=dims)
    den = preds.sum(dim=dims) + targets.sum(dim=dims)
    return ((num + eps) / (den + eps)).mean()


def dice_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    den = pred.sum() + gt.sum()
    return float((2 * inter + eps) / (den + eps))


def iou_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + eps) / (union + eps))


def precision_recall_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> Tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return float((tp + eps) / (tp + fp + eps)), float((tp + eps) / (tp + fn + eps))


# -----------------------------
# Training / validation
# -----------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, device: str) -> Dict[str, float]:
    model.train()
    total_loss, total_dice = 0.0, 0.0
    for batch in tqdm(loader, desc="Train"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            dsc = dice_score(logits, masks)
        total_loss += loss.item()
        total_dice += dsc.item()
    return {"loss": total_loss / len(loader), "dice": total_dice / len(loader)}


@torch.no_grad()
def validate_one_epoch(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    for batch in tqdm(loader, desc="Val"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
        dsc = dice_score(logits, masks)
        total_loss += loss.item()
        total_dice += dsc.item()
    return {"loss": total_loss / len(loader), "dice": total_dice / len(loader)}


def plot_training_history(history: list[Dict[str, float]], output_path: Path) -> None:
    df = pd.DataFrame(history)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["train_dice"], label="train")
    plt.plot(df["epoch"], df["val_dice"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Dice")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def checkpoint_payload(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: list[Dict[str, float]],
    best_val_dice: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_val_dice": best_val_dice,
        "args": vars(args),
    }


def save_training_artifacts(history: list[Dict[str, float]], run_dir: Path) -> None:
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    with open(run_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_training_history(history, run_dir / "plots" / "training_history.png")


def train(args: argparse.Namespace, run_dir: Path, device: str) -> nn.Module:
    dict_pairs = create_image_mask_pairs(args.path_volumes, args.path_masks, args.path_ids_link_file)
    train_pairs, val_pairs, test_pairs = split_dict_pairs(dict_pairs, args.train_size, args.val_size, args.test_size, args.seed)
    with open(run_dir / "data_split.json", "w") as f:
        json.dump({"train": list(train_pairs), "val": list(val_pairs), "test": list(test_pairs)}, f, indent=2)
    print(f"Split sizes: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

    train_ds = NoduleVideoDataset(
        train_pairs,
        image_size=args.image_size,
        num_frames=args.num_frames,
        use_2p5d=args.use_2p5d,
        positive_prob=args.positive_prob,
        min_fg_pixels=args.min_fg_pixels,
        window_level=args.window_level,
        window_width=args.window_width,
        random_window=args.random_window,
        allow_negative_volumes=args.allow_negative_volumes,
    )
    val_ds = NoduleVideoDataset(
        val_pairs,
        image_size=args.image_size,
        num_frames=args.num_frames,
        use_2p5d=args.use_2p5d,
        positive_prob=args.positive_prob,
        min_fg_pixels=args.min_fg_pixels,
        window_level=args.window_level,
        window_width=args.window_width,
        random_window=False,
        allow_negative_volumes=args.allow_negative_volumes,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=args.val_num_workers, pin_memory=args.pin_memory)

    model = build_model(args, device)
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {num_trainable:,}")
    print(f"Total params:     {num_total:,}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    history: list[Dict[str, float]] = []
    best_val_dice = -1.0
    ckpt_dir = run_dir / "checkpoints"

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_metrics = validate_one_epoch(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        save_training_artifacts(history, run_dir)

        print(
            f"Epoch {epoch:03d}/{args.num_epochs} | "
            f"train loss {row['train_loss']:.4f} | train dice {row['train_dice']:.4f} | "
            f"val loss {row['val_loss']:.4f} | val dice {row['val_dice']:.4f}"
        )

        payload = checkpoint_payload(epoch, model, optimizer, history, best_val_dice, args)
        latest_path = ckpt_dir / "latest_checkpoint.pt"
        torch.save(payload, latest_path)
        torch.save(model.state_dict(), ckpt_dir / "latest_weights.pt")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            payload = checkpoint_payload(epoch, model, optimizer, history, best_val_dice, args)
            improved_ckpt = ckpt_dir / f"improved_epoch{epoch:03d}_valdice{best_val_dice:.5f}.pt"
            improved_weights = ckpt_dir / f"improved_weights_epoch{epoch:03d}_valdice{best_val_dice:.5f}.pt"
            best_ckpt = ckpt_dir / "best_checkpoint.pt"
            best_weights = ckpt_dir / "best_weights.pt"
            torch.save(payload, improved_ckpt)
            torch.save(model.state_dict(), improved_weights)
            shutil.copyfile(improved_ckpt, best_ckpt)
            shutil.copyfile(improved_weights, best_weights)
            print(f"Saved improved and best checkpoints: val_dice={best_val_dice:.5f}")

    if args.evaluate_test:
        test_ds = NoduleVideoDataset(
            test_pairs,
            image_size=args.image_size,
            num_frames=args.num_frames,
            use_2p5d=args.use_2p5d,
            positive_prob=args.positive_prob,
            min_fg_pixels=args.min_fg_pixels,
            window_level=args.window_level,
            window_width=args.window_width,
            random_window=False,
            allow_negative_volumes=args.allow_negative_volumes,
        )
        test_loader = DataLoader(test_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=args.val_num_workers, pin_memory=args.pin_memory)
        test_metrics = evaluate_test(model, test_loader, device, threshold=args.threshold)
        with open(run_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Test mean Dice: {test_metrics['mean_dice']:.5f}")

    return model


# -----------------------------
# Optional clip-level test eval
# -----------------------------

@torch.no_grad()
def evaluate_test(model: nn.Module, loader: DataLoader, device: str, threshold: float = 0.5) -> Dict[str, Any]:
    model.eval()
    dices = []
    for batch in tqdm(loader, desc="Test"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        dims = tuple(range(1, preds.ndim))
        num = 2 * (preds * masks).sum(dim=dims)
        den = preds.sum(dim=dims) + masks.sum(dim=dims)
        dice = ((num + 1e-6) / (den + 1e-6))
        dices.extend(float(x) for x in dice.detach().cpu().tolist())
    return {"mean_dice": float(sum(dices) / len(dices)), "all_dice": dices}


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune prompt-free SAM2 on CT nodule clips.")

    # Paths
    parser.add_argument("--path-dataset", default="/datasets/LUNA_dataset/", help="Dataset root.")
    parser.add_argument("--path-volumes", default=None, help="Folder containing .mhd CT volumes. Defaults to PATH_DATASET/CT_volumes.")
    parser.add_argument("--path-masks", default=None, help="Folder containing nodule masks. Defaults to PATH_DATASET/masks_nodules/nifti_data.")
    parser.add_argument("--path-ids-link-file", default=None, help="CSV mapping SeriesID to CID. Defaults to PATH_DATASET/LUNA16_metadata_split_offical.csv.")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-checkpoint", default="../checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")

    # Data and preprocessing
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--use-2p5d", type=str2bool, default=False)
    parser.add_argument("--positive-prob", type=float, default=0.7)
    parser.add_argument("--min-fg-pixels", type=int, default=5)
    parser.add_argument("--window-level", type=float, default=-600)
    parser.add_argument("--window-width", type=float, default=1500)
    parser.add_argument("--random-window", type=str2bool, default=True)
    parser.add_argument("--allow-negative-volumes", type=str2bool, default=True)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--test-size", type=float, default=0.10)

    # Training
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", type=str2bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    # Trainable parts
    parser.add_argument("--train-mask-decoder", type=str2bool, default=True)
    parser.add_argument("--train-prompt-encoder", type=str2bool, default=False)
    parser.add_argument("--train-image-encoder", type=str2bool, default=False)

    # Eval
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    dataset = Path(args.path_dataset)
    if args.path_volumes is None:
        args.path_volumes = str(dataset / "CT_volumes")
    if args.path_masks is None:
        args.path_masks = str(dataset / "masks_nodules" / "nifti_data")
    if args.path_ids_link_file is None:
        args.path_ids_link_file = str(dataset / "LUNA16_metadata_split_offical.csv")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    print(f"Using device: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    run_dir = make_run_dir(args)
    save_config(args, run_dir)
    print(f"Run directory: {run_dir}")
    train(args, run_dir, device)


if __name__ == "__main__":
    main()
