#!/usr/bin/env python3
"""Convert ISPRS segmentation masks to COCO JSON format for SAM 3 training."""

import os, json, cv2, numpy as np
from PIL import Image
from datetime import datetime

# 5 classes (no clutter)
CLASSES = ['road', 'building', 'grass', 'tree', 'car']
# ISPRS label mapping: 1=road, 2=building, 3=grass, 4=tree, 5=car, 6=clutter(ignored)
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

DATASETS = {
    'vaihingen': {
        'data_root': '/root/autodl-tmp/dataset/Vaihingen',
        'img_dir': 'top',
        'gt_dir': 'gts_index',
        'img_suffix': '.tif',
        'gt_suffix': '.png',
        'tiles': [
            'top_mosaic_09cm_area1', 'top_mosaic_09cm_area3', 'top_mosaic_09cm_area5',
            'top_mosaic_09cm_area7', 'top_mosaic_09cm_area11', 'top_mosaic_09cm_area13',
            'top_mosaic_09cm_area15', 'top_mosaic_09cm_area17', 'top_mosaic_09cm_area21',
            'top_mosaic_09cm_area23', 'top_mosaic_09cm_area26', 'top_mosaic_09cm_area28',
        ],
    },
    'potsdam': {
        'data_root': '/root/autodl-tmp/dataset/Potsdam',
        'img_dir': '2_Ortho_RGB',
        'gt_dir': 'labels_index',
        'img_suffix': '_RGB.tif',
        'gt_suffix': '.png',
        'tiles': [
            'top_potsdam_2_10', 'top_potsdam_2_11', 'top_potsdam_2_12',
            'top_potsdam_3_10', 'top_potsdam_3_11', 'top_potsdam_3_12',
            'top_potsdam_4_10', 'top_potsdam_4_11', 'top_potsdam_4_12',
            'top_potsdam_5_10', 'top_potsdam_5_11', 'top_potsdam_5_12',
            'top_potsdam_6_10', 'top_potsdam_6_11', 'top_potsdam_6_12',
            'top_potsdam_7_10', 'top_potsdam_7_11', 'top_potsdam_7_12',
        ],
    },
}


def mask_to_polygons(mask):
    """Convert binary mask to list of polygons (simplified)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        # Simplify and convert to list
        cnt = cnt.squeeze(1)
        if cnt.ndim == 1:
            continue
        poly = cnt.ravel().tolist()
        if len(poly) >= 6:  # at least 3 points (x,y pairs)
            polygons.append(poly)
    return polygons


def extract_instances(gt_path):
    """Extract instance polygons per class from segmentation mask."""
    gt = np.array(Image.open(gt_path))
    instances = []

    for orig_label, cls_idx in LABEL_MAP.items():
        cls_mask = (gt == orig_label).astype(np.uint8)
        if cls_mask.sum() < 50:  # skip tiny regions
            continue

        polygons = mask_to_polygons(cls_mask)
        for poly in polygons:
            # Compute bbox from polygon
            xs = poly[0::2]
            ys = poly[1::2]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            w, h = xmax - xmin, ymax - ymin
            if w < 5 or h < 5:
                continue

            instances.append({
                'category_id': cls_idx,
                'bbox': [float(xmin), float(ymin), float(w), float(h)],
                'segmentation': [poly],
                'area': float(cls_mask.sum()),
                'iscrowd': 0,
            })

    return instances


def create_coco_json(dataset_name, output_dir):
    """Create COCO JSON for a dataset."""
    info = DATASETS[dataset_name]
    data_root = info['data_root']
    os.makedirs(output_dir, exist_ok=True)

    images = []
    annotations = []
    ann_id = 0

    for img_id, tile in enumerate(info['tiles'], 1):
        img_path = os.path.join(data_root, info['img_dir'], f'{tile}{info["img_suffix"]}')
        gt_path = os.path.join(data_root, info['gt_dir'], f'{tile}{info["gt_suffix"]}')

        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            print(f"  SKIP {tile}: missing files")
            continue

        img = Image.open(img_path)
        w, h = img.size

        img_filename = f'{tile}.jpg'
        img_out = os.path.join(output_dir, img_filename)
        if not os.path.exists(img_out):
            img.convert('RGB').save(img_out, quality=95)

        images.append({
            'id': img_id,
            'file_name': img_filename,
            'width': w,
            'height': h,
        })

        instances = extract_instances(gt_path)
        for inst in instances:
            ann_id += 1
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': inst['category_id'] + 1,  # COCO uses 1-indexed categories
                'bbox': inst['bbox'],
                'segmentation': inst['segmentation'],
                'area': inst['area'],
                'iscrowd': inst['iscrowd'],
            })

        if img_id % 5 == 0:
            print(f"  Processed {img_id}/{len(info['tiles'])}: {tile}")

    categories = [{'id': i+1, 'name': name, 'supercategory': 'object'}
                   for i, name in enumerate(CLASSES)]

    coco = {
        'info': {'description': f'ISPRS {dataset_name} for SAM3 training',
                  'date_created': datetime.now().isoformat()},
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }

    json_path = os.path.join(output_dir, 'annotations.json')
    with open(json_path, 'w') as f:
        json.dump(coco, f)
    print(f"  Saved: {json_path} ({len(images)} images, {len(annotations)} annotations)")


def main():
    for ds_name in ['vaihingen', 'potsdam']:
        print(f"\n{'='*60}")
        print(f"Converting {ds_name}...")
        output_dir = f'/root/Mynet/sam3_isprs/{ds_name}'
        create_coco_json(ds_name, output_dir)

    # Also create combined dataset
    print(f"\n{'='*60}")
    print("Creating combined dataset...")
    output_dir = '/root/Mynet/sam3_isprs/combined'
    os.makedirs(output_dir, exist_ok=True)

    all_images = []
    all_anns = []
    ann_offset = 0
    img_offset = 0

    for ds_name in ['vaihingen', 'potsdam']:
        json_path = f'/root/Mynet/sam3_isprs/{ds_name}/annotations.json'
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            data = json.load(f)

        for img in data['images']:
            img['id'] += img_offset
            old_name = img['file_name']
            new_name = f'{ds_name}_{old_name}'
            img['file_name'] = new_name
            # Copy image
            src = f'/root/Mynet/sam3_isprs/{ds_name}/{old_name}'
            dst = f'{output_dir}/{new_name}'
            if os.path.exists(src) and not os.path.exists(dst):
                os.system(f'cp "{src}" "{dst}"')

        for ann in data['annotations']:
            ann['id'] += ann_offset
            ann['image_id'] += img_offset

        all_images.extend(data['images'])
        all_anns.extend(data['annotations'])
        img_offset += len(data['images'])
        ann_offset += len(data['annotations'])

    coco = {
        'info': {'description': 'ISPRS Combined Vaihingen+Potsdam'},
        'images': all_images,
        'annotations': all_anns,
        'categories': [{'id': i+1, 'name': n, 'supercategory': 'object'} for i,n in enumerate(CLASSES)],
    }
    json.dump(coco, open(f'{output_dir}/annotations.json', 'w'))
    print(f"  Combined: {len(all_images)} images, {len(all_anns)} annotations")


if __name__ == '__main__':
    main()
