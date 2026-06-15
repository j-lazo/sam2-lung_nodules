import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import pandas as pd
import os
import SimpleITK as sitk
from matplotlib import pyplot as plt

import sys
import tqdm
import seaborn as sns
import random
import math
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

def normalize_to_uint8(img_2d):
    img_2d = img_2d.astype(np.float32)
    img_2d = img_2d - img_2d.min()
    max_val = img_2d.max()
    if max_val > 0:
        img_2d = img_2d / max_val
    return (img_2d * 255).astype(np.uint8)


def dice_score(mask1, mask2, smooth=1e-6):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()

    return float((2 * intersection + smooth) / (total + smooth))


def normalize_to_uint8(img_2d):
    img_2d = img_2d.astype(np.float32)
    img_2d = img_2d - img_2d.min()
    max_val = img_2d.max()
    if max_val > 0:
        img_2d = img_2d / max_val
    return (img_2d * 255).astype(np.uint8)


def dice_score(mask1, mask2, smooth=1e-6):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()

    return float((2 * intersection + smooth) / (total + smooth))


def build_2p5d_slice(image_array, z):
    """
    image_array: [Z, Y, X]
    returns RGB-like [H, W, 3] using [z-1, z, z+1]
    """
    z_prev = max(z - 1, 0)
    z_next = min(z + 1, image_array.shape[0] - 1)

    prev_slice = normalize_to_uint8(image_array[z_prev])
    curr_slice = normalize_to_uint8(image_array[z])
    next_slice = normalize_to_uint8(image_array[z_next])

    img_3ch = np.stack([prev_slice, curr_slice, next_slice], axis=-1)
    return img_3ch


def compute_mask_shape_features(mask):
    """
    mask: binary [H, W]
    """
    mask = (mask > 0).astype(np.uint8)
    area = int(mask.sum())
    if area == 0:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)

    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = float(w) / float(h) if h > 0 else np.inf
    elongation = max(aspect_ratio, 1.0 / max(aspect_ratio, 1e-8))

    if perimeter > 0:
        circularity = 4.0 * math.pi * area / (perimeter ** 2)
    else:
        circularity = 0.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
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
    mask,
    min_area=5,
    max_area=400,
    min_circularity=0.20,
    min_solidity=0.70,
    max_elongation=2.0,
):
    feats = compute_mask_shape_features(mask)
    if feats is None:
        return False, None

    keep = (
        feats["area"] >= min_area and
        feats["area"] <= max_area and
        feats["circularity"] >= min_circularity and
        feats["solidity"] >= min_solidity and
        feats["elongation"] <= max_elongation
    )
    return keep, feats


def merge_sam2_automatic_masks_blob_like(
    anns,
    image_shape,
    min_area=5,
    max_area=400,
    min_circularity=0.20,
    min_solidity=0.70,
    max_elongation=2.0,
    pred_iou_thresh=None,
    stability_score_thresh=None,
    return_kept_anns=False,
):
    """
    anns: list of dicts from SAM2AutomaticMaskGenerator.generate(...)
    """
    H, W = image_shape
    merged_mask = np.zeros((H, W), dtype=bool)
    kept_anns = []

    for ann in anns:
        seg = ann["segmentation"].astype(np.uint8)

        pred_iou = ann.get("predicted_iou", None)
        stability = ann.get("stability_score", None)

        if pred_iou_thresh is not None and pred_iou is not None and pred_iou < pred_iou_thresh:
            continue
        if stability_score_thresh is not None and stability is not None and stability < stability_score_thresh:
            continue

        keep, feats = is_nodule_like_mask(
            seg,
            min_area=min_area,
            max_area=max_area,
            min_circularity=min_circularity,
            min_solidity=min_solidity,
            max_elongation=max_elongation,
        )

        if not keep:
            continue

        merged_mask |= seg.astype(bool)

        if return_kept_anns:
            ann_copy = dict(ann)
            ann_copy["shape_features"] = feats
            kept_anns.append(ann_copy)

    if return_kept_anns:
        return merged_mask.astype(np.uint8), kept_anns

    return merged_mask.astype(np.uint8)


def analyze_CT_volume_sam2_automatic(mhd_file, path_volumes, path_masks, df_ids_link, mask_generator, list_number_masks, list_maks_nodules, process_all_slices=False, min_area=5,
                                     max_area=400, min_circularity=0.20, min_solidity=0.70, max_elongation=2.0, pred_iou_thresh=None, stability_score_thresh=None,):
    
    path_image = os.path.join(path_volumes, mhd_file + ".mhd")
    if not os.path.isfile(path_image):
        print("Missing image:", path_image)
        return None

    sub_df_links = df_ids_link[df_ids_link["SeriesID"] == mhd_file]
    if len(sub_df_links) == 0:
        print("No mask link found for:", mhd_file)
        return None

    mask_id_num = sub_df_links["CID"].tolist()[0]
    if mask_id_num not in list_number_masks:
        print("Mask id not found:", mask_id_num, mhd_file)
        return None

    idx = list_number_masks.index(mask_id_num)
    path_mask = os.path.join(path_masks, list_maks_nodules[idx])
    if not os.path.isfile(path_mask):
        print("Missing mask:", path_mask)
        return None

    image = sitk.ReadImage(path_image)
    image_array = sitk.GetArrayFromImage(image)   # [Z, Y, X]

    mask = sitk.ReadImage(path_mask)
    gt_volume = sitk.GetArrayFromImage(mask)
    gt_volume = (gt_volume >= 0.5).astype(np.uint8)

    pred_volume = np.zeros_like(gt_volume, dtype=np.uint8)

    if process_all_slices:
        z_list = range(image_array.shape[0])
    else:
        z_list = np.where(gt_volume.reshape(gt_volume.shape[0], -1).sum(axis=1) > 0)[0]

    for z in z_list:
        img_3ch = build_2p5d_slice(image_array, z)

        anns = mask_generator.generate(img_3ch)

        pred_slice = merge_sam2_automatic_masks_blob_like(anns=anns, image_shape=img_3ch.shape[:2], min_area=min_area, max_area=max_area,
                                                          min_circularity=min_circularity, min_solidity=min_solidity, max_elongation=max_elongation,
                                                          pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh,)

        pred_volume[z] = pred_slice

    dsc_vol = dice_score(gt_volume, pred_volume)

    return {
        "seriesuid": mhd_file,
        "image_array":image_array,
        "dice_volume": dsc_vol,
        "gt_volume": gt_volume,
        "pred_volume": pred_volume,
    }


def show_volume_slice_result(image_array, gt_volume, pred_volume, z):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_array[z], cmap="gray")
    plt.title(f"CT slice {z}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_volume[z], cmap="gray")
    plt.title("GT")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_volume[z], cmap="gray")
    plt.title("SAM2 automatic")
    plt.axis("off")

    plt.show()


def build_slice_prompt_dict(mask_3d):
    """
    Group prompts by z-slice.

    Parameters
    ----------
    mask_3d : np.ndarray
        Binary mask with shape [Z, Y, X]

    Returns
    -------
    slice_dicts : list of dict
        One dict per z-slice containing at least one blob.

        Each dict contains:
        - z
        - mask_slice
        - blobs
        - point_coords: np.ndarray of shape (N, 2)
        - point_labels: np.ndarray of shape (N,)
        - boxes: np.ndarray of shape (N, 4)
    """
    mask_3d = (mask_3d > 0).astype(np.uint8)
    slice_dicts = []

    for z in range(mask_3d.shape[0]):
        mask_slice = mask_3d[z]

        if not np.any(mask_slice > 0):
            continue

        blobs = extract_blobs_from_slice(mask_slice)

        if len(blobs) == 0:
            continue

        point_coords = np.array([blob["center"] for blob in blobs], dtype=np.float32)   # (N, 2)
        point_labels = np.ones(len(blobs), dtype=np.int32)                              # (N,)
        boxes = np.array([blob["bbox"] for blob in blobs], dtype=np.float32)            # (N, 4)

        slice_dicts.append({
            "z": z,
            "mask_slice": mask_slice,
            "blobs": blobs,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "boxes": boxes,
        })

    return slice_dicts


def get_contours_and_centroids(mask_2d):
    """
    Extract every contour/blob from a binary 2D mask and compute its centroid.

    Parameters
    ----------
    mask_2d : np.ndarray
        Binary mask [H, W]

    Returns
    -------
    blobs : list of dict
        Each dict contains:
        - contour
        - centroid : [x, y]
        - bbox : [x_min, y_min, x_max, y_max]
        - component_mask : binary mask for this contour/blob
    """
    mask_bin = (mask_2d > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []

    for contour in contours:
        if contour.shape[0] == 0:
            continue

        M = cv2.moments(contour)

        # Handle degenerate contour
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            # fallback: mean of contour points
            pts = contour[:, 0, :]
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])

        x, y, w, h = cv2.boundingRect(contour)

        component_mask = np.zeros_like(mask_bin, dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], contourIdx=-1, color=1, thickness=-1)

        blobs.append({
            "contour": contour,
            "centroid": [float(cx), float(cy)],
            "bbox": [int(x), int(y), int(x + w - 1), int(y + h - 1)],
            "component_mask": component_mask
        })

    return blobs

def get_all_blobs_from_mask_3d(mask_3d):
    """
    Go through all slices in a 3D mask [Z, Y, X].
    For every non-empty slice, find all contours/blobs.

    Returns
    -------
    all_blobs : list of dict
        Each dict contains:
        - z
        - centroid
        - bbox
        - component_mask
        - contour
    """
    mask_3d = (mask_3d > 0).astype(np.uint8)
    all_blobs = []

    for z in range(mask_3d.shape[0]):
        mask_slice = mask_3d[z]

        if np.any(mask_slice > 0):
            blobs = get_contours_and_centroids(mask_slice)

            for blob in blobs:
                all_blobs.append({
                    "z": z,
                    "centroid": blob["centroid"],
                    "bbox": blob["bbox"],
                    "component_mask": blob["component_mask"],
                    "contour": blob["contour"]
                })

    return all_blobs


def get_interior_point(component_mask):
    """
    Return [x, y] point inside the component, good for SAM prompting.
    """
    component_mask = (component_mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return [float(x), float(y)]

def extract_blobs_from_slice(mask_2d):
    """
    For one binary slice [H, W], extract all blobs/contours.

    Returns
    -------
    blobs : list of dict
        Each dict has:
        - center: [x, y]
        - bbox: [x_min, y_min, x_max, y_max]
        - component_mask: [H, W]
        - contour
    """
    mask_bin = (mask_2d > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []

    for contour in contours:
        if contour is None or len(contour) == 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        component_mask = np.zeros_like(mask_bin, dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], contourIdx=-1, color=1, thickness=-1)

        if component_mask.sum() == 0:
            continue

        center_xy = get_interior_point(component_mask)

        blobs.append({
            "center": center_xy,
            "bbox": [int(x), int(y), int(x + w - 1), int(y + h - 1)],
            "component_mask": component_mask,
            "contour": contour,
        })

    return blobs


def build_sam2_input_slice(image_array, z, use_triplet_channels=False):
    """
    image_array: [Z, Y, X]
    returns [H, W, 3] uint8
    """
    if use_triplet_channels:
        z_prev = max(z - 1, 0)
        z_next = min(z + 1, image_array.shape[0] - 1)

        ch0 = normalize_to_uint8(image_array[z_prev])
        ch1 = normalize_to_uint8(image_array[z])
        ch2 = normalize_to_uint8(image_array[z_next])

        img_3ch = np.stack([ch0, ch1, ch2], axis=-1)
    else:
        img = normalize_to_uint8(image_array[z])
        img_3ch = np.stack([img, img, img], axis=-1)

    return img_3ch


def analyze_CT_volume_sam2_slice_wise_all_modes(
    mhd_file,
    path_volumes,
    path_masks,
    df_ids_link,
    predictor,                 # SAM2ImagePredictor
    mask_generator,            # SAM2AutomaticMaskGenerator
    list_number_masks,
    list_maks_nodules,
    use_triplet_channels=False,
    automatic_process_all_slices=False,
    automatic_min_area=5,
    automatic_max_area=400,
    automatic_min_circularity=0.20,
    automatic_min_solidity=0.70,
    automatic_max_elongation=2.0,
    automatic_pred_iou_thresh=None,
    automatic_stability_score_thresh=None,
    plot=False):
    
    path_image = os.path.join(path_volumes, mhd_file + ".mhd")

    if not os.path.isfile(path_image):
        print("Missing image:", path_image)
        return None

    image = sitk.ReadImage(path_image)
    image_array = sitk.GetArrayFromImage(image)   # [Z, Y, X]

    sub_df_links = df_ids_link[df_ids_link["SeriesID"] == mhd_file]
    if len(sub_df_links) == 0:
        print("No mask link found for:", mhd_file)
        return None

    mask_id_num = sub_df_links["CID"].tolist()[0]
    if mask_id_num not in list_number_masks:
        print("Mask id not found:", mask_id_num, mhd_file)
        return None

    idx = list_number_masks.index(mask_id_num)
    path_mask = os.path.join(path_masks, list_maks_nodules[idx])

    if not os.path.isfile(path_mask):
        print("Missing mask:", path_mask)
        return None

    mask = sitk.ReadImage(path_mask)
    mask_array = sitk.GetArrayFromImage(mask)     # [Z, Y, X]
    gt_volume = (mask_array >= 0.5).astype(np.uint8)

    slice_prompt_dicts = build_slice_prompt_dict(gt_volume)
    if len(slice_prompt_dicts) == 0:
        print("No valid slices found for:", mhd_file)
        return None

    # prompt-based branch only needs slices with GT foreground
    prompt_slice_dict = {d["z"]: d for d in slice_prompt_dicts}

    # automatic branch can be all slices or only GT-positive slices
    if automatic_process_all_slices:
        automatic_z_list = list(range(image_array.shape[0]))
    else:
        automatic_z_list = sorted(prompt_slice_dict.keys())

    all_z_to_process = sorted(set(prompt_slice_dict.keys()).union(automatic_z_list))

    # Full 3D prediction volumes
    pred_auto_vol = np.zeros_like(gt_volume, dtype=bool)
    pred_point_vol = np.zeros_like(gt_volume, dtype=bool)
    pred_box_vol = np.zeros_like(gt_volume, dtype=bool)
    pred_combined_vol = np.zeros_like(gt_volume, dtype=bool)

    # Optional score storage for prompted modes
    all_scores_points = []
    all_scores_boxes = []
    all_scores_combined = []

    for z in all_z_to_process:
        img_3ch_automatic = build_sam2_input_slice(image_array=image_array, z=z, use_triplet_channels=use_triplet_channels,)
        img_3ch = build_sam2_input_slice(image_array=image_array, z=z, use_triplet_channels=False,)

        # ---------------------------------
        # 1) Automatic SAM2 on this slice
        # ---------------------------------
        if z in automatic_z_list:
            anns = mask_generator.generate(img_3ch_automatic)

            pred_auto_slice = merge_sam2_automatic_masks_blob_like(
                anns=anns,
                image_shape=img_3ch_automatic.shape[:2],
                min_area=automatic_min_area,
                max_area=automatic_max_area,
                min_circularity=automatic_min_circularity,
                min_solidity=automatic_min_solidity,
                max_elongation=automatic_max_elongation,
                pred_iou_thresh=automatic_pred_iou_thresh,
                stability_score_thresh=automatic_stability_score_thresh,)

            pred_auto_vol[z] |= pred_auto_slice.astype(bool)


        # ---------------------------------
        # 2) Prompted SAM2 on this slice
        # ---------------------------------
        if z in prompt_slice_dict:
            slice_info = prompt_slice_dict[z]
            point_coords = slice_info["point_coords"]   # (N, 2)
            point_labels = slice_info["point_labels"]   # (N,)
            boxes = slice_info["boxes"]                 # (N, 4)
            blobs = slice_info["blobs"]

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(img_3ch)

                pred_point_slice = np.zeros_like(gt_volume[z], dtype=bool)
                pred_box_slice = np.zeros_like(gt_volume[z], dtype=bool)
                pred_combined_slice = np.zeros_like(gt_volume[z], dtype=bool)

                for i in range(len(blobs)):
                    input_point = point_coords[i:i+1]   # (1, 2)
                    input_label = point_labels[i:i+1]   # (1,)
                    input_box = boxes[i:i+1]            # (1, 4)

                    # point
                    masks_points, scores_points, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False,)
                    # box
                    masks_boxes, scores_boxes, _ = predictor.predict(point_coords=None, point_labels=None, box=input_box, multimask_output=False,)
                    # point + box
                    masks_combined, scores_combined, _ = predictor.predict(point_coords=input_point, point_labels=input_label, box=input_box, multimask_output=False,)

                    pred_point_slice |= masks_points[0].astype(bool)
                    pred_box_slice |= masks_boxes[0].astype(bool)
                    pred_combined_slice |= masks_combined[0].astype(bool)

                    all_scores_points.append(float(scores_points[0]))
                    all_scores_boxes.append(float(scores_boxes[0]))
                    all_scores_combined.append(float(scores_combined[0]))

                pred_point_vol[z] |= pred_point_slice
                pred_box_vol[z] |= pred_box_slice
                pred_combined_vol[z] |= pred_combined_slice


            if plot: 
                plt.figure()
                plt.subplot(1,6,1)
                plt.imshow(img_3ch)
                plt.subplot(1,6,2)
                plt.imshow(gt_volume[z, :, :])
                plt.subplot(1,6,3)
                plt.imshow(pred_auto_slice)
                plt.subplot(1,6,4)
                plt.imshow(masks_points[0, :, :])
                plt.subplot(1,6,5)
                plt.imshow(masks_boxes[0, :, :])
                plt.subplot(1,6,6)
                plt.imshow(masks_combined[0, :, :])

    # Convert to uint8
    pred_auto_vol = pred_auto_vol.astype(np.uint8)
    pred_point_vol = pred_point_vol.astype(np.uint8)
    pred_box_vol = pred_box_vol.astype(np.uint8)
    pred_combined_vol = pred_combined_vol.astype(np.uint8)

    # Whole-volume Dice
    dsc_auto_vol = dice_score(gt_volume, pred_auto_vol)
    dsc_point_vol = dice_score(gt_volume, pred_point_vol)
    dsc_box_vol = dice_score(gt_volume, pred_box_vol)
    dsc_combined_vol = dice_score(gt_volume, pred_combined_vol)

    return (mhd_file, dsc_auto_vol, dsc_point_vol, dsc_box_vol, dsc_combined_vol, all_scores_points, all_scores_boxes, all_scores_combined, pred_auto_vol, pred_point_vol, pred_box_vol, pred_combined_vol, gt_volume,)


path_dataset = '/datasets/LUNA_dataset/'
list_files_dataset = os.listdir(path_dataset)
annoations_path = os.path.join(path_dataset, 'annotations.csv')
df_annotations = pd.read_csv(annoations_path)

unique_series_uids, repetitions_uids = np.unique(df_annotations['seriesuid'].tolist(), return_counts=True)
print('Unique IDs with labels: ', len(unique_series_uids), 'Note: One ID could have more than one set of annotations (i.e. more than 1 nodule)')


path_ct_volumes = os.path.join(path_dataset, 'CT_volumes')
list_all_files = list()
#sub_folders_ct_volues = os.listdir(path_ct_volumes)

list_all_files += os.listdir(path_ct_volumes)

annotations_name_list = df_annotations['seriesuid'].tolist()
mhd_files = [f for f in list_all_files if f.endswith('.mhd')]
only_name_files = [f.replace('.mhd', '') for f in mhd_files]
print(len(only_name_files), 'CT Volumes in ', path_ct_volumes)

# masks analysis 
path_masks = '/datasets/LUNA_dataset/masks_nodules/nifti_data/'
list_files_path = os.listdir(path_masks)
list_maks_nodules = [f for f in list_files_path if 'mask' in f and 'contour' in f and 'circle' not in f and 'nodule' in f]
print(len(list_maks_nodules), 'Masks in ', path_masks)
list_number_masks = [int(s.split('_')[0]) for s in list_maks_nodules]

path_annotaions_links = '/mimer/NOBACKUP/groups/naiss2025-6-383/jorge/datasets/LUNA_dataset/LUNA16_metadata_split_offical.csv'
df_ids_link = pd.read_csv(path_annotaions_links)

# run this in case you want to get the cases in which there are more than 1 nodule per CT Volume 

dup_rows = df_annotations[df_annotations["seriesuid"].duplicated(keep=False)]["seriesuid"].tolist()


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
    
    
#sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
base_path = 'projects/repos/sam2/checkpoints'
sam2_checkpoint = os.path.join(base_path, 'sam2.1_hiera_large.pt')
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
print(os.path.isfile(model_cfg))
print(os.path.isfile(sam2_checkpoint))

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)
mask_generator = SAM2AutomaticMaskGenerator(sam2_model)


path_volumes = os.path.join(path_dataset, 'CT_volumes')

list_dsc_auto = []
list_dsc_point = []
list_dsc_box = []
list_dsc_combined = []
file_names_list = []

for mhd_file in tqdm.tqdm(only_name_files):
    result = analyze_CT_volume_sam2_slice_wise_all_modes(
        mhd_file=mhd_file,
        path_volumes=path_volumes,
        path_masks=path_masks,
        df_ids_link=df_ids_link,
        predictor=predictor,
        mask_generator=mask_generator,
        list_number_masks=list_number_masks,
        list_maks_nodules=list_maks_nodules,
        use_triplet_channels=True,
        automatic_process_all_slices=False,
        automatic_min_area=5,
        automatic_max_area=400,
        automatic_min_circularity=0.20,
        automatic_min_solidity=0.70,
        automatic_max_elongation=2.0,
        automatic_pred_iou_thresh=0.7,
        automatic_stability_score_thresh=0.8,)

    if result is not None:
        
        img_name, dsc_auto_vol, dsc_point_vol, dsc_box_vol, dsc_combined_vol, *_ = result
    
        list_dsc_auto.append(dsc_auto_vol)
        list_dsc_point.append(dsc_point_vol)
        list_dsc_box.append(dsc_box_vol)
        list_dsc_combined.append(dsc_combined_vol)
        file_names_list.append(img_name)


df_results = pd.DataFrame({'VolumeID': file_names_list, 'DSC (image points)': list_dsc_point,
                           'DSC (image boxes)': list_dsc_box, 'DSC (image combines)': list_dsc_combined, 
                           'DSC (image auto.)': list_dsc_auto,})


path_save_results = os.path.join(path_dataset, 'predictions_SAM2_images_single_ch.csv')
df_results.to_csv(path_save_results, index=False)
print(f'results saved at {path_save_results}')
