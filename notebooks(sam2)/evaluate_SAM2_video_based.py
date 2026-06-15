import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from scipy import ndimage

import pandas as pd
import tqdm
import seaborn as sns
import random
from matplotlib import pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


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


def build_rgb_frame_from_volume(volume_zyx, z, use_triplet_channels=False):
    """
    volume_zyx: [Z, Y, X]
    returns RGB uint8 frame [H, W, 3]
    """
    Z = volume_zyx.shape[0]

    if use_triplet_channels:
        z_prev = max(z - 1, 0)
        z_next = min(z + 1, Z - 1)

        ch0 = normalize_to_uint8(volume_zyx[z_prev])
        ch1 = normalize_to_uint8(volume_zyx[z])
        ch2 = normalize_to_uint8(volume_zyx[z_next])
        rgb = np.stack([ch0, ch1, ch2], axis=-1)
    else:
        img = normalize_to_uint8(volume_zyx[z])
        rgb = np.stack([img, img, img], axis=-1)

    return rgb


def write_volume_as_sam2_frames(volume_zyx, out_dir, use_triplet_channels=False, ext=".jpg"):
    """
    Saves slices as 00000.jpg, 00001.jpg, ... for SAM2 video predictor.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for z in range(volume_zyx.shape[0]):
        rgb = build_rgb_frame_from_volume(volume_zyx, z, use_triplet_channels=use_triplet_channels)
        Image.fromarray(rgb).save(out_dir / f"{z:05d}{ext}")
        

def get_3d_connected_components(mask_zyx):
    """
    mask_zyx: binary [Z, Y, X]
    returns labeled volume and number of components
    """
    structure = ndimage.generate_binary_structure(rank=3, connectivity=2)
    labeled, num = ndimage.label(mask_zyx.astype(np.uint8), structure=structure)
    return labeled, num


def get_component_center_point_3d(component_mask):
    """
    component_mask: binary [Z, Y, X] for one object
    returns point as (z, y, x), chosen as distance-transform peak
    """
    dist = ndimage.distance_transform_edt(component_mask)
    zyx = np.unravel_index(np.argmax(dist), dist.shape)
    return zyx  # (z, y, x)


def get_bbox_from_2d_mask(mask_2d):
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def extract_nodule_prompts_from_3d_mask(mask_zyx):
    """
    Returns one prompt dict per 3D connected nodule.
    """
    labeled, num = get_3d_connected_components(mask_zyx)
    prompts = []

    obj_id = 1
    for k in range(1, num + 1):
        comp = (labeled == k)
        if comp.sum() == 0:
            continue

        z, y, x = get_component_center_point_3d(comp)

        center_slice_mask = comp[z].astype(np.uint8)
        box_xyxy = get_bbox_from_2d_mask(center_slice_mask)
        if box_xyxy is None:
            continue

        prompts.append({
            "obj_id": obj_id,
            "center_zyx": (int(z), int(y), int(x)),
            "frame_idx": int(z),
            "point_xy": np.array([[float(x), float(y)]], dtype=np.float32),  # (1,2)
            "point_labels": np.array([1], dtype=np.int32),                    # (1,)
            "box_xyxy": box_xyxy,                                             # (4,)
            "component_mask": comp.astype(np.uint8),
        })
        obj_id += 1

    return prompts


# build SAM Video Predictor: 


def build_sam2_video_model(model_cfg, checkpoint, device="cuda", vos_optimized=False):
    predictor = build_sam2_video_predictor(
        model_cfg,
        checkpoint,
        device=device,
        vos_optimized=vos_optimized,)
    
    return predictor


def build_sam2_video_model_from_hf(model_name="facebook/sam2-hiera-large"):
    predictor = SAM2VideoPredictor.from_pretrained(model_name)
    return predictor


def add_sam2_prompt_for_object(predictor, inference_state, frame_idx, obj_id, prompt_mode, point_xy=None, point_labels=None, box_xyxy=None,):
    """
    prompt_mode: 'point', 'box', or 'point+box'
    """
    if prompt_mode == "point":
        if point_xy is None or point_labels is None:
            raise ValueError("point mode requires point_xy and point_labels")

        return predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=point_xy,
            labels=point_labels,
        )

    elif prompt_mode == "box":
        if box_xyxy is None:
            raise ValueError("box mode requires box_xyxy")

        return predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box_xyxy,
        )

    elif prompt_mode == "point+box":
        if point_xy is None or point_labels is None or box_xyxy is None:
            raise ValueError("point+box mode requires point_xy, point_labels, and box_xyxy")

        return predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=point_xy,
            labels=point_labels,
            box=box_xyxy,
        )

    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


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
    plt.title("SAM2 prediction")
    plt.axis("off")

    plt.show()


def analyze_CT_volume_sam2(
    mhd_file,
    path_volumes,
    path_masks,
    df_ids_link,
    predictor,
    list_number_masks,
    list_maks_nodules,
    prompt_mode="point",           # 'point', 'box', 'point+box'
    use_triplet_channels=False,
    cleanup_frames=True,
):
    """
    Whole-volume SAM2 evaluation on a CT volume treated as a video.

    Returns
    -------
    dict or None
    """
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

    prompts = extract_nodule_prompts_from_3d_mask(gt_volume)
    if len(prompts) == 0:
        print("No 3D nodules found in mask:", mhd_file)
        return None

    temp_dir = tempfile.mkdtemp(prefix=f"sam2_{mhd_file}_")

    try:
        write_volume_as_sam2_frames(image_array, temp_dir, use_triplet_channels=use_triplet_channels, ext=".jpg",)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = predictor.init_state(video_path=temp_dir)

            if hasattr(predictor, "reset_state"):
                predictor.reset_state(inference_state)

            # Register one prompt per nodule
            for p in prompts:
                add_sam2_prompt_for_object(predictor=predictor, inference_state=inference_state, frame_idx=p["frame_idx"], obj_id=p["obj_id"],
                                           prompt_mode=prompt_mode, point_xy=p["point_xy"], point_labels=p["point_labels"], box_xyxy=p["box_xyxy"],)

            Z, H, W = gt_volume.shape
            pred_by_obj = {p["obj_id"]: np.zeros((Z, H, W), dtype=np.uint8)for p in prompts}

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                for i, obj_id in enumerate(out_obj_ids):
                    mask_logits = out_mask_logits[i]
                    mask_np = mask_logits.squeeze().float().cpu().numpy()
                    mask_bin = (mask_np > 0.0).astype(np.uint8)
                    pred_by_obj[int(obj_id)][int(out_frame_idx)] = mask_bin

        pred_volume = np.zeros_like(gt_volume, dtype=np.uint8)
        for obj_id, obj_mask in pred_by_obj.items():
            pred_volume = np.logical_or(pred_volume, obj_mask).astype(np.uint8)

        dsc_vol = dice_score(gt_volume, pred_volume)

        return {
            "seriesuid": mhd_file,
            "image_array":image_array,
            "prompt_mode": prompt_mode,
            "dice_volume": dsc_vol,
            "gt_volume": gt_volume,
            "pred_volume": pred_volume,
            "prompts": prompts,
            "pred_by_obj": pred_by_obj,}

    finally:
        if cleanup_frames:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    path_dataset = '/mimer/NOBACKUP/groups/naiss2025-6-383/jorge/datasets/LUNA_dataset/'
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
    path_masks = '/mimer/NOBACKUP/groups/naiss2025-6-383/jorge/datasets/LUNA_dataset/masks_nodules/nifti_data/'
    list_files_path = os.listdir(path_masks)
    list_maks_nodules = [f for f in list_files_path if 'mask' in f and 'contour' in f and 'circle' not in f and 'nodule' in f]
    print(len(list_maks_nodules), 'Masks in ', path_masks)
    list_number_masks = [int(s.split('_')[0]) for s in list_maks_nodules]
    
    path_annotaions_links = '/mimer/NOBACKUP/groups/naiss2025-6-383/jorge/datasets/LUNA_dataset/LUNA16_metadata_split_offical.csv'
    df_ids_link = pd.read_csv(path_annotaions_links)
    
    # run this in case you want to get the cases in which there are more than 1 nodule per CT Volume 
    path_volumes = path_ct_volumes
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
    base_path = '/mimer/NOBACKUP/groups/naiss2025-6-383/jorge/projects/repos/sam2/checkpoints'
    sam2_checkpoint = os.path.join(base_path, 'sam2.1_hiera_large.pt')
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    print(os.path.isfile(model_cfg))
    print(os.path.isfile(sam2_checkpoint))
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


    results = []

    for mhd_file in tqdm.tqdm(only_name_files):
        try:
            res_point = analyze_CT_volume_sam2(
                mhd_file=mhd_file,
                path_volumes=path_volumes,
                path_masks=path_masks,
                df_ids_link=df_ids_link,
                predictor=predictor,
                list_number_masks=list_number_masks,
                list_maks_nodules=list_maks_nodules,
                prompt_mode="point",)
    
            res_box = analyze_CT_volume_sam2(
                mhd_file=mhd_file,
                path_volumes=path_volumes,
                path_masks=path_masks,
                df_ids_link=df_ids_link,
                predictor=predictor,
                list_number_masks=list_number_masks,
                list_maks_nodules=list_maks_nodules,
                prompt_mode="box",)
    
            res_point_box = analyze_CT_volume_sam2(
                mhd_file=mhd_file,
                path_volumes=path_volumes,
                path_masks=path_masks,
                df_ids_link=df_ids_link,
                predictor=predictor,
                list_number_masks=list_number_masks,
                list_maks_nodules=list_maks_nodules,
                prompt_mode="point+box",)
    
            row = {
                "VolumeID": mhd_file,
                "DSC (video points)": None if res_point is None else res_point.get("dice_volume"),
                "DSC (video boxes)": None if res_box is None else res_box.get("dice_volume"),
                "DSC (video combines)": None if res_point_box is None else res_point_box.get("dice_volume"),
                "missing_point": res_point is None,
                "missing_box": res_box is None,
                "missing_point_box": res_point_box is None,
            }
    
            results.append(row)
    
        except Exception as e:
            results.append({
                "VolumeID": mhd_file,
                "DSC (video points)": None,
                "DSC (video boxes)": None,
                "DSC (video combines)": None,
                "missing_point": True,
                "missing_box": True,
                "missing_point_box": True,
                "error": str(e),
            })
    
    df_results = pd.DataFrame(results)

    path_save_results = os.path.join(path_dataset, "predictions_SAM2_video.csv")
    df_results.to_csv(path_save_results, index=False)
    print(f"results saved at: {path_save_results}")

    print("Mean SAM2 point-only Dice:", df_results["DSC (video points)"].mean())
    print("Mean SAM2 box-only Dice:", df_results["DSC (video boxes)"].mean())
    print("Mean SAM2 point+box Dice:", df_results["DSC (video combines)"].mean())
    

if __name__ == "__main__":
    main()