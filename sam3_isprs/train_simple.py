#!/usr/bin/env python3
"""
Simplified SAM 3 training for ISPRS data — 绕过 Hydra/submitit 复杂度

使用 SAM 3 官方的 model_builder 和 loss 函数，配合自制的 COCO 数据集训练。
"""

import os, sys, json, time, gc, argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pycocotools import mask as mask_util

# Setup paths — sam3-main needs to be importable as 'sam3'
sys.path.insert(0, '/root/Mynet/sam3-main')
sys.path.insert(0, '/root/Mynet/sam3-main/sam3')
sys.path.insert(0, '/root/Mynet/SegEarth-OV-3-main')

# Patch so 'sam3.train' is importable from the source tree
import types
if 'sam3' not in sys.modules:
    sam3_pkg = types.ModuleType('sam3')
    sam3_pkg.__path__ = ['/root/Mynet/sam3-main/sam3']
    sys.modules['sam3'] = sam3_pkg

from model_builder import build_sam3_image_model
from train.data.coco_json_loaders import load_coco_and_group_by_image
from train.loss.sam3_loss import Sam3LossWrapper
from train.matcher import BinaryHungarianMatcherV2
from train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, PadToSizeAPI, ToTensorAPI, NormalizeAPI
from train.transforms.point_sampling import RandomizeInputBbox
from train.transforms.segmentation import DecodeRle
from train.transforms.filter_query_transforms import FlexibleFilterFindGetQueries, FilterCrowds, FilterEmptyTargets

# Simplified combined dataset for ISPRS
CLASSES = ['road', 'building', 'grass', 'tree', 'car']


def collate_fn(batch):
    """Simple collate: filter None and stack."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    images = torch.stack([b['image'] for b in batch])
    targets = [b['target'] for b in batch]
    return {'image': images, 'target': targets}


class ISPRSDataset(torch.utils.data.Dataset):
    """Load COCO JSON and provide image + annotations for SAM3 training."""

    def __init__(self, json_path, img_dir, resolution=1008, is_train=True):
        grouped, self.cat_id_to_name = load_coco_and_group_by_image(json_path)
        self.samples = grouped
        self.img_dir = img_dir
        self.resolution = resolution
        self.is_train = is_train

        if is_train:
            self.transforms = ComposeAPI(transforms=[
                RandomizeInputBbox(box_noise_std=0.1, box_noise_max=20),
                DecodeRle(),
                RandomResizeAPI(sizes=[resolution], max_size=[resolution], square=True, consistent_transform=False),
                PadToSizeAPI(size=resolution, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])
        else:
            self.transforms = ComposeAPI(transforms=[
                DecodeRle(),
                RandomResizeAPI(sizes=[resolution], max_size=[resolution], square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_info = sample['image']
        anns = sample['annotations']

        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            return None
        from PIL import Image
        image = Image.open(img_path).convert('RGB')

        # Build target dict
        target = {
            'image_id': img_info['id'],
            'annotations': anns,
            'image': np.array(image),
        }

        try:
            result = self.transforms(target)
            return result
        except Exception:
            return None


def train_one_epoch(model, dataloader, optimizer, scaler, loss_fn, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue

        images = batch['image'].to(device)
        targets = batch['target']

        # Move targets to device
        for t in targets:
            for k, v in t.items():
                if isinstance(v, torch.Tensor):
                    t[k] = v.to(device)

        optimizer.zero_grad()

        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(images, targets)
            loss_dict = loss_fn(outputs, targets)
            loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"  E{epoch}/{total_epochs} B{batch_idx}/{len(dataloader)} loss={loss.item():.4f}", end='\r')

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='combined', choices=['vaihingen', 'potsdam', 'combined'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output-dir', default='/root/Mynet/autodl-tmp/runs')
    args = parser.parse_args()

    device = torch.device('cuda')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'sam3_official_{args.dataset}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    data_dir = f'/root/Mynet/sam3_isprs/{args.dataset}'
    json_path = f'{data_dir}/annotations.json'

    print(f"Loading SAM 3 model...")
    model = build_sam3_image_model(
        bpe_path='/root/Mynet/SegEarth-OV-3-main/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path='/root/Mynet/SegEarth-OV-3-main/weights/sam3/sam3.pt',
        device='cuda', eval_mode=False, enable_segmentation=True,
    )

    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable:,} trainable / {total:,} total")

    # Logger
    print(f"Loading dataset from {json_path}...")
    dataset = ISPRSDataset(json_path, data_dir, resolution=1008, is_train=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=2, collate_fn=collate_fn, drop_last=True)
    print(f"  {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # Optimizer, loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # SAM3 loss (detection-style)
    matcher = BinaryHungarianMatcherV2(focal=True, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
                                        alpha=0.25, gamma=2, stable=False)
    loss_fn = Sam3LossWrapper(matcher=matcher, o2m_weight=2.0)

    print(f"\nTraining ({args.epochs} epochs)...")
    history = {'loss': []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, dataloader, optimizer, scaler, loss_fn, device, epoch, args.epochs)
        scheduler.step()
        elapsed = time.time() - t0
        history['loss'].append(avg_loss)
        print(f"  Epoch {epoch:3d}/{args.epochs}: loss={avg_loss:.4f}  ({elapsed:.0f}s)")

        # Save checkpoint
        if epoch % 2 == 0:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'history': history,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt'))

    # Final save
    torch.save({
        'epoch': args.epochs, 'model_state_dict': model.state_dict(),
        'history': history,
    }, os.path.join(output_dir, 'final_model.pt'))
    json.dump({'mode': 'official_sam3', 'dataset': args.dataset,
               'epochs': args.epochs, 'history': history},
              open(os.path.join(output_dir, 'config.json'), 'w'), indent=2)

    print(f"\nSaved to: {output_dir}")
    del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
