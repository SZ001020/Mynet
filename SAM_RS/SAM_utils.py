import os
import re
import time

import numpy as np
import torch
from PIL import Image
from skimage.segmentation import find_boundaries

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def build_mask_generator(model_type, checkpoint, device, crop_nms_thresh, box_nms_thresh, pred_iou_thresh):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        sam,
        crop_nms_thresh=crop_nms_thresh,
        box_nms_thresh=box_nms_thresh,
        pred_iou_thresh=pred_iou_thresh,
    )


def sam_aug(image, mask_generator, min_area=50, max_instances=50):
    masks = mask_generator.generate(image)
    h, w = image.shape[:2]
    if len(masks) == 0:
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    boundary_prior = np.zeros((h, w), dtype=np.uint8)
    object_map = np.zeros((h, w), dtype=np.uint8)

    sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    instance_id = 1
    for ann in sorted_masks:
        if ann["area"] < min_area:
            continue
        if instance_id > max_instances:
            break
        m = ann["segmentation"]
        object_map[m] = instance_id
        instance_id += 1

    for ann in masks:
        thismask = ann["segmentation"]
        mask_ = np.zeros(thismask.shape, dtype=np.uint8)
        mask_[thismask] = 1
        boundary_prior = np.maximum(boundary_prior, find_boundaries(mask_, mode="thick").astype(np.uint8))

    boundary_prior[boundary_prior > 0] = 255
    object_map[boundary_prior > 0] = 0
    return boundary_prior.astype(np.uint8), object_map.astype(np.uint8)


def parse_tile_id(filename):
    # Vaihingen: top_mosaic_09cm_area1.tif -> 1
    m = re.search(r"area(.+?)\.tif$", filename)
    if m:
        return m.group(1)

    # Potsdam: top_potsdam_6_10_RGBIR.tif / top_potsdam_6_10_RGB.tif -> 6_10
    m = re.search(r"top_potsdam_(.+?)_(RGBIR|RGB)\.tif$", filename)
    if m:
        return m.group(1)

    return None


def main():
    # Defaults target MFNet training expectations on this workspace.
    input_dir = os.environ.get("SSRS_SAM_INPUT_DIR", "/root/Mynet/autodl-tmp/dataset/Vaihingen/top")
    boundary_out = os.environ.get("SSRS_SAM_BOUNDARY_OUT", "/root/Mynet/autodl-tmp/dataset/Vaihingen/sam_boundary_merge")
    object_out = os.environ.get("SSRS_SAM_OBJECT_OUT", "/root/Mynet/autodl-tmp/dataset/Vaihingen/V_merge")
    object_prefix = os.environ.get("SSRS_SAM_OBJECT_PREFIX", "V")

    model_type = os.environ.get("SSRS_SAM_MODEL_TYPE", "vit_h")
    checkpoint = os.environ.get("SSRS_SAM_CKPT", "./sam_vit_h_4b8939.pth")
    device = os.environ.get("SSRS_SAM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    crop_nms_thresh = float(os.environ.get("SSRS_SAM_CROP_NMS", "0.5"))
    box_nms_thresh = float(os.environ.get("SSRS_SAM_BOX_NMS", "0.5"))
    pred_iou_thresh = float(os.environ.get("SSRS_SAM_PRED_IOU", "0.96"))
    min_area = int(os.environ.get("SSRS_SAM_MIN_AREA", "50"))
    max_instances = int(os.environ.get("SSRS_SAM_MAX_INSTANCES", "50"))

    print("PyTorch version:", torch.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    print("Input dir:", input_dir)
    print("Boundary out:", boundary_out)
    print("Object out:", object_out)
    print("Object prefix:", object_prefix)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    os.makedirs(boundary_out, exist_ok=True)
    os.makedirs(object_out, exist_ok=True)

    mask_generator = build_mask_generator(
        model_type=model_type,
        checkpoint=checkpoint,
        device=device,
        crop_nms_thresh=crop_nms_thresh,
        box_nms_thresh=box_nms_thresh,
        pred_iou_thresh=pred_iou_thresh,
    )

    img_list = sorted([f for f in os.listdir(input_dir) if f.endswith(".tif")])
    if len(img_list) == 0:
        raise RuntimeError(f"No .tif files found in {input_dir}")

    start_time = time.time()
    ok_count = 0
    skip_count = 0

    for img_input in img_list:
        img_id = parse_tile_id(img_input)
        if img_id is None:
            print(f"[Skip] Cannot parse tile id from filename: {img_input}")
            skip_count += 1
            continue

        image = np.array(Image.open(os.path.join(input_dir, img_input)))
        boundary_map, object_map = sam_aug(
            image,
            mask_generator,
            min_area=min_area,
            max_instances=max_instances,
        )

        boundary_path = os.path.join(boundary_out, f"ISPRS_merge_{img_id}.tif")
        object_path = os.path.join(object_out, f"{object_prefix}_merge_{img_id}.tif")

        Image.fromarray(boundary_map).save(boundary_path)
        Image.fromarray(object_map).save(object_path)
        ok_count += 1

        if ok_count % 5 == 0:
            print(f"[Progress] processed {ok_count}/{len(img_list)}")

    run_time = time.time() - start_time
    print(f"Done. processed={ok_count}, skipped={skip_count}, elapsed_sec={run_time:.2f}")


if __name__ == "__main__":
    main()