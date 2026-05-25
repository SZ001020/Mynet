#!/usr/bin/env python3
"""Plan3 Route A: Ground Truth vs Fine-tuned 推理对比可视化"""

import os, sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SE = '/root/Mynet/Reference-Project/SegEarth-OV-3-main'
sys.path.insert(0, SE); sys.path.insert(0, '/root/Mynet/Personal-Project/RS-SAM3-p3')

from adapter_unet import AdapterSAM3UNet
from dataset_adapter import _rgb_to_class, VAIHINGEN_VAL

PALETTE = [[255,255,255],[0,0,255],[0,255,255],[0,255,0],[255,255,0]]
CLASS_NAMES = ['road', 'building', 'grass', 'tree', 'car']
IGNORE_COLOR = [255,0,0]

def pred_to_color(pred):
    c = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        c[pred == i] = color
    c[pred == 255] = IGNORE_COLOR
    return c

def load_model(ckpt_path):
    _prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    sam3 = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device='cuda')
    os.chdir(_prev)
    model = AdapterSAM3UNet(sam3, adapter_bottleneck=32, num_classes=5).cuda()
    model.resolution = 1008
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model, ckpt

def main():
    ckpt_path = '/root/Mynet/autodl-tmp/runs/plan3_adapter_20260501_224049/best_model.pt'
    out_dir = '/root/Mynet/autodl-tmp/runs/plan3_adapter_20260501_224049'

    print("Loading model...")
    model, ckpt = load_model(ckpt_path)
    print(f"  Epoch {ckpt['epoch']}, mIoU={ckpt['best_v']:.1f}%")

    tiles = VAIHINGEN_VAL
    img_dir = '/root/autodl-tmp/dataset/Vaihingen/top'
    gt_dir = '/root/autodl-tmp/dataset/Vaihingen/gts_for_participants'

    for tile in tiles:
        ip = f'{img_dir}/{tile}.tif'; gp = f'{gt_dir}/{tile}.tif'
        if not os.path.exists(ip): continue

        img_full = np.array(Image.open(ip).convert('RGB'))
        gt_rgb = np.array(Image.open(gp).convert('RGB'))
        gt_idx = _rgb_to_class(gt_rgb)

        # Center crop 512x512
        h, w = img_full.shape[:2]
        cy, cx = h//2, w//2
        y1, y2 = cy-256, cy+256; x1, x2 = cx-256, cx+256
        img_crop = img_full[y1:y2, x1:x2]
        gt_crop = gt_idx[y1:y2, x1:x2]

        # Inference
        img_t = torch.from_numpy(img_crop).permute(2,0,1).float().unsqueeze(0)/255.0
        with torch.no_grad():
            logits = model(img_t.cuda())
            logits = F.interpolate(logits, (512,512), mode='bilinear', align_corners=False)
            pred = logits.argmax(1)[0].cpu().numpy()

        # Make sure shapes match
        assert pred.shape == gt_crop.shape, f"{pred.shape} vs {gt_crop.shape}"

        mask = gt_crop != 255
        oa = (pred[mask] == gt_crop[mask]).sum() / mask.sum() * 100

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img_crop)
        axes[0].set_title('Input (center crop)', fontsize=14); axes[0].axis('off')
        axes[1].imshow(pred_to_color(gt_crop))
        axes[1].set_title('Ground Truth', fontsize=14); axes[1].axis('off')
        axes[2].imshow(pred_to_color(pred))
        axes[2].set_title(f'Fine-tuned (OA={oa:.1f}%)', fontsize=14); axes[2].axis('off')

        legend = [plt.Rectangle((0,0),1,1, fc=np.array(c)/255) for c in PALETTE]
        fig.legend(legend, CLASS_NAMES, loc='lower center', ncol=5, fontsize=10, frameon=False)
        fig.savefig(f'{out_dir}/compare_{tile}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {tile} OA={oa:.1f}%")

    del model; torch.cuda.empty_cache()
    print(f"\nDone! → {out_dir}/")

if __name__ == '__main__':
    main()
