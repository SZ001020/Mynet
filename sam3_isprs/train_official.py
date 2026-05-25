#!/usr/bin/env python3
"""
Phase 3/4: SAM 3 官方训练模式 — 直接在 ISPRS COCO 数据上微调

核心: 使用 build_sam3_image_model(eval_mode=False) 进入训练模式，
     配合 simplified detection loss 进行 fine-tuning。
"""

import os, sys, json, time, gc, argparse
from datetime import datetime
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Paths — use sam3-main, NOT SegEarth's older sam3 copy
sys.path.insert(0, '/root/Mynet/sam3-main')
SE = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, SE)
# Remove SegEarth/sam3 from search to avoid import conflicts
for p in list(sys.path):
    if p == os.path.join(SE, 'sam3'):
        sys.path.remove(p)


# ============================================================
# Inline COCO loader (avoid complex sam3 package imports)
# ============================================================
def load_coco_and_group_by_image(json_path):
    with open(json_path) as f:
        coco = json.load(f)
    images = {img['id']: img for img in coco['images']}
    anns_by_image = defaultdict(list)
    for ann in coco['annotations']:
        anns_by_image[ann['image_id']].append(ann)
    grouped = []
    for image_id in sorted(images.keys()):
        grouped.append({'image': images[image_id], 'annotations': anns_by_image.get(image_id, [])})
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}
    return grouped, cat_id_to_name


# Now import SAM3 model (from sam3-main, not SegEarth)
from sam3.model_builder import build_sam3_image_model


# ============================================================
# Dataset
# ============================================================

class ISPRSDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, img_dir, resolution=1008):
        grouped, self.cat_id_to_name = load_coco_and_group_by_image(json_path)
        self.samples = grouped
        self.img_dir = img_dir
        self.resolution = resolution

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_info = s['image']
        anns = s['annotations']

        img_path = os.path.join(self.img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            return None

        image = Image.open(img_path).convert('RGB')
        W, H = image.size

        # Resize to 1008x1008
        ratio = self.resolution / max(W, H)
        new_w, new_h = int(W * ratio), int(H * ratio)
        image = image.resize((new_w, new_h), Image.BILINEAR)

        # Convert annotations to normalized bbox format
        boxes = []
        class_ids = []
        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h] in absolute coords
            x, y, bw, bh = bbox
            # Convert to [cx, cy, w, h] normalized
            cx = (x + bw / 2) / W
            cy = (y + bh / 2) / H
            nw = bw / W
            nh = bh / H
            if nw > 0.001 and nh > 0.001:
                boxes.append([cx, cy, nw, nh])
                class_ids.append(ann['category_id'] - 1)  # 1-indexed → 0-indexed

        # To tensor
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        # Pad to 1008x1008
        _, h, w = img_tensor.shape
        if h < self.resolution or w < self.resolution:
            pad_h = max(0, self.resolution - h)
            pad_w = max(0, self.resolution - w)
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), value=0)

        # Normalize to [-1, 1]
        img_tensor = (img_tensor - 0.5) / 0.5

        return {
            'image': img_tensor,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            'labels': torch.tensor(class_ids, dtype=torch.long) if class_ids else torch.zeros(0, dtype=torch.long),
            'orig_size': (H, W),
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    images = torch.stack([b['image'] for b in batch])
    targets = [{
        'boxes': b['boxes'], 'labels': b['labels'], 'orig_size': b['orig_size'],
    } for b in batch]
    return images, targets


# ============================================================
# Simple Detection Loss
# ============================================================

def simple_detection_loss(outputs, targets):
    """Simplified loss: use SAM3's pred_logits and pred_boxes with CE + L1."""
    pred_logits = outputs.get('pred_logits')  # (B, N_queries, num_classes)
    pred_boxes = outputs.get('pred_boxes')     # (B, N_queries, 4)

    if pred_logits is None or pred_boxes is None:
        return torch.tensor(0.0, requires_grad=True)

    B, N, C = pred_logits.shape
    if pred_logits.dim() == 4:
        pred_logits = pred_logits.mean(1)  # handle aux output
        pred_boxes = pred_boxes.mean(1) if pred_boxes.dim() == 4 else pred_boxes

    total_loss = 0.0
    num_targets = 0

    for b in range(B):
        tgt_boxes = targets[b]['boxes'].to(pred_boxes.device)
        tgt_labels = targets[b]['labels'].to(pred_logits.device)
        if len(tgt_boxes) == 0:
            continue

        # Simple matching: nearest query to each target
        pred_b = pred_boxes[b]  # (N, 4)
        logits_b = pred_logits[b]  # (N, C)

        box_loss = 0.0
        cls_loss = 0.0
        for i in range(min(len(tgt_boxes), pred_b.shape[0])):
            # Use first N query slots directly
            box_loss += F.l1_loss(pred_b[i], tgt_boxes[i])
            cls_loss += F.cross_entropy(logits_b[i:i+1], tgt_labels[i:i+1])

        total_loss += (box_loss + cls_loss) / max(len(tgt_boxes), 1)
        num_targets += 1

    return total_loss / max(num_targets, 1)


# ============================================================
# Training
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='combined', choices=['vaihingen','potsdam','combined'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--output-dir', default='/root/Mynet/autodl-tmp/runs')
    args = parser.parse_args()

    device = torch.device('cuda')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'sam3_official_{args.dataset}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    data_dir = f'/root/Mynet/sam3_isprs/{args.dataset}'
    json_path = f'{data_dir}/annotations.json'

    print(f"Loading SAM 3 (training mode)...")
    model = build_sam3_image_model(
        bpe_path='/root/Mynet/SegEarth-OV-3-main/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path='/root/Mynet/SegEarth-OV-3-main/weights/sam3/sam3.pt',
        device='cuda', eval_mode=False, enable_segmentation=True,
    )
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {t:,} / {total:,} trainable")

    print(f"Loading data: {json_path}")
    dataset = ISPRSDataset(json_path, data_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2,
                                          collate_fn=collate_fn, drop_last=True)
    print(f"  {len(dataset)} images, {len(loader)} batches/epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = {'loss': []}

    print(f"\nTraining {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for bi, (images, targets) in enumerate(loader):
            if images is None:
                continue
            images = images.to(device)

            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                # Forward through SAM3
                backbone_out = model.backbone.forward_image(images)
                batch_data = BatchedDatapoint(tensor=images, targets=targets,
                                              image_sizes=[t['orig_size'] for t in targets])
                outputs = model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=model._get_dummy_find(),
                    geometric_prompt=model._get_dummy_prompt(),
                    find_target=batch_data,
                )
                loss = simple_detection_loss(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if bi % 10 == 0:
                print(f"  E{epoch}/{args.epochs} B{bi}/{len(loader)} loss={loss.item():.4f}", end='\r')

        scheduler.step()
        avg_loss = epoch_loss / max(len(loader), 1)
        history['loss'].append(avg_loss)
        print(f"  Epoch {epoch}/{args.epochs}: loss={avg_loss:.4f}  ({time.time()-t0:.0f}s)")

        if epoch % 2 == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'history': history},
                       os.path.join(output_dir, f'ckpt_epoch{epoch}.pt'))

    torch.save({'epoch': args.epochs, 'model': model.state_dict(), 'history': history},
               os.path.join(output_dir, 'final_model.pt'))
    json.dump({'mode': 'official_sam3', 'dataset': args.dataset, 'history': history},
              open(os.path.join(output_dir, 'config.json'), 'w'), indent=2)
    print(f"\nDone: {output_dir}")
    del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
