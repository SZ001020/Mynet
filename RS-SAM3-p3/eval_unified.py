#!/usr/bin/env python3
"""
统一 256² 滑动窗口评估脚本。
支持 Plan3 (AdapterSAM3UNetFormer/DSM) 和 Plan4 (SAM3FullTrain) 架构。

用法:
  cd /root/Mynet
  # Plan4 Full
  python RS-SAM3-p3/eval_unified.py --ckpt /root/autodl-tmp/runs/plan4_full_vaihingen_20260507_175600/best_model.pt --arch plan4_full --dataset vaihingen
  # Plan3 MFNetDec DSM
  python RS-SAM3-p3/eval_unified.py --ckpt /root/autodl-tmp/runs/plan3_mfnetdec_dsm_20260506_220419/best_model.pt --arch plan3_dsm --dataset vaihingen
"""

import os, sys, json, argparse, time
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image

SE = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, SE)
sys.path.insert(0, '/root/Mynet/RS-SAM3-p3')
sys.path.insert(0, '/root/Mynet/RS-SAM3-p3r')
sys.path.insert(0, '/root/Mynet/RS-SAM3-p4')

from dataset_adapter import _rgb_to_class, VAIHINGEN_VAL, POTSDAM_VAL

CLASS_NAMES = ['road', 'building', 'grass', 'tree', 'car']


def load_model_plan3_rgb(ckpt_path, device='cuda'):
    """Plan3 AdapterSAM3UNetFormer (RGB only)"""
    from adapter_unet import AdapterSAM3UNetFormer
    _prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    sam3 = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device=device)
    os.chdir(_prev)
    model = AdapterSAM3UNetFormer(sam3, adapter_bottleneck=32, num_classes=5, dropout=0.1).to(device)
    model.resolution = 1008
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Remap old SeparableConvBN: old=[dwconv,BN,pwconv] → new=[dwconv,pwconv,BN]
    state = ckpt['model']
    for blk in ['b2', 'b3', 'b4', 'p1']:
        prefix = f'decoder.{blk}.attn.proj'
        k1 = [k for k in state if k.startswith(f'{prefix}.1.')]
        k2 = [k for k in state if k.startswith(f'{prefix}.2.')]
        for k in k1:
            state[k.replace('.attn.proj.1.', '.attn.proj._tmp_.')] = state.pop(k)
        for k in k2:
            state[k.replace('.attn.proj.2.', '.attn.proj.1.')] = state.pop(k)
        for k in list(state.keys()):
            if '.attn.proj._tmp_.' in k:
                state[k.replace('.attn.proj._tmp_.', '.attn.proj.2.')] = state.pop(k)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ckpt


def load_model_plan3_dsm(ckpt_path, device='cuda', decode_channels=64):
    """Plan3 AdapterSAM3UNetFormerDSM (RGB+DSM)"""
    from adapter_unet import AdapterSAM3UNetFormerDSM
    _prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    sam3 = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device=device)
    os.chdir(_prev)
    model = AdapterSAM3UNetFormerDSM(sam3, adapter_bottleneck=32, num_classes=5,
                                     decode_channels=decode_channels, dropout=0.1).to(device)
    model.resolution = 1008
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Remap old SeparableConvBN: old=[dwconv,BN,pwconv] → new=[dwconv,pwconv,BN]
    # Proj.1 (old BN) ↔ proj.2 (old pwconv)
    state = ckpt['model']
    remap_blocks = ['b2', 'b3', 'b4', 'p1']
    for blk in remap_blocks:
        prefix = f'decoder.{blk}.attn.proj'
        # Collect old keys
        k1 = [k for k in state if k.startswith(f'{prefix}.1.')]
        k2 = [k for k in state if k.startswith(f'{prefix}.2.')]
        # Temp rename: 1→tmp, 2→1, tmp→2
        for k in k1:
            state[k.replace('.attn.proj.1.', '.attn.proj._tmp_.')] = state.pop(k)
        for k in k2:
            state[k.replace('.attn.proj.2.', '.attn.proj.1.')] = state.pop(k)
        for k in list(state.keys()):
            if '.attn.proj._tmp_.' in k:
                state[k.replace('.attn.proj._tmp_.', '.attn.proj.2.')] = state.pop(k)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ckpt


def load_model_plan4_full(ckpt_path, device='cuda'):
    """Plan4 SAM3FullTrain (RGB+DSM, full ViT training)"""
    from train_full import SAM3FullTrain
    _prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    sam3 = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device=device)
    os.chdir(_prev)
    model = SAM3FullTrain(sam3, num_classes=5, use_dsm=True, dropout=0.1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_patch_rgb(model, patch_256, arch):
    t = torch.from_numpy(patch_256).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t = F.interpolate(t, (1008, 1008), mode='bilinear', align_corners=False)
    logits = model(t.cuda())
    logits = F.interpolate(logits, (256, 256), mode='bilinear', align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()


@torch.no_grad()
def predict_patch_dsm(model, patch_rgb_256, patch_dsm_256, arch):
    t_rgb = torch.from_numpy(patch_rgb_256).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t_dsm = torch.from_numpy(patch_dsm_256).float().unsqueeze(0)
    t_rgb = F.interpolate(t_rgb, (1008, 1008), mode='bilinear', align_corners=False)
    t_dsm = F.interpolate(t_dsm.unsqueeze(1), (1008, 1008), mode='bilinear', align_corners=False)
    logits = model(t_rgb.cuda(), t_dsm.cuda())
    logits = F.interpolate(logits, (256, 256), mode='bilinear', align_corners=False)
    return logits.argmax(1)[0].cpu().numpy()


def sliding_256(tile_h, tile_w, model, img_np, dsm_np, arch, stride=128):
    pred_sum = np.zeros((tile_h, tile_w), dtype=np.float64)
    count = np.zeros((tile_h, tile_w), dtype=np.float64)

    for y in range(0, tile_h - 128, stride):
        for x in range(0, tile_w - 128, stride):
            y2, x2 = min(y + 256, tile_h), min(x + 256, tile_w)
            ph, pw = y2 - y, x2 - x
            if ph < 128 or pw < 128:
                continue

            patch_rgb = img_np[y:y2, x:x2]
            if dsm_np is not None:
                patch_dsm = dsm_np[y:y2, x:x2]
                pred = predict_patch_dsm(model, patch_rgb, patch_dsm, arch)
            else:
                pred = predict_patch_rgb(model, patch_rgb, arch)

            m = 16
            im, jm = min(m, ph // 4), min(m, pw // 4)
            pred_sum[y + im:y2 - im, x + jm:x2 - jm] += pred[im:ph - im, jm:pw - jm]
            count[y + im:y2 - im, x + jm:x2 - jm] += 1.0

    count[count == 0] = 1.0
    return np.round(pred_sum / count).astype(np.int64)


def evaluate_tile(model, tile, img_dir, gt_dir, dsm_dir, img_suf, gt_suf,
                  dsm_stem_fn, use_dsm, arch):
    ip = f'{img_dir}/{tile}{img_suf}'
    gp = f'{gt_dir}/{tile}{gt_suf}'
    img = np.array(Image.open(ip).convert('RGB'))
    gt = _rgb_to_class(np.array(Image.open(gp).convert('RGB')))

    dsm_np = None
    if use_dsm:
        dp = f'{dsm_dir}/dsm_{dsm_stem_fn(tile)}'
        if os.path.exists(dp):
            dsm_np = np.array(Image.open(dp)).astype(np.float32)
            dsm_np = (dsm_np - dsm_np.min()) / max(dsm_np.max() - dsm_np.min(), 1e-8)

    t0 = time.time()
    pred = sliding_256(img.shape[0], img.shape[1], model, img, dsm_np, arch)
    elapsed = time.time() - t0

    mask = gt != 255
    total = mask.sum()
    oa = (pred[mask] == gt[mask]).sum() / max(total, 1) * 100
    ious = {}
    oas = {}
    for c in range(5):
        pc, lc = pred == c, gt == c
        inter = (pc & lc).sum()
        union = (pc | lc).sum()
        ious[CLASS_NAMES[c]] = float(inter / union * 100) if union > 0 else 0.0
        tn = ((~pc) & (~lc) & mask).sum()
        oas[CLASS_NAMES[c]] = float((inter + tn) / max(total, 1) * 100)
    miou = np.mean(list(ious.values()))
    moa = np.mean(list(oas.values()))
    print(f"  {tile} ({img.shape[1]}x{img.shape[0]}) [{elapsed:.0f}s]: "
          f"OA={oa:.1f}% mIoU={miou:.1f}% per-cls-OA={moa:.1f}%")
    return oa, miou, ious, oas


def main():
    ap = argparse.ArgumentParser(description='Unified 256² sliding window eval')
    ap.add_argument('--ckpt', required=True, help='Checkpoint path')
    ap.add_argument('--arch', required=True,
                    choices=['plan3_rgb', 'plan3_dsm', 'plan4_full'],
                    help='Model architecture')
    ap.add_argument('--dataset', default='vaihingen', choices=['vaihingen', 'potsdam'])
    ap.add_argument('--output', default=None, help='Output dir (default: ckpt dir)')
    ap.add_argument('--decode-channels', type=int, default=64,
                    help='Decoder channels (64 for older checkpoints, 256 for newer)')
    args = ap.parse_args()

    use_dsm = args.arch in ('plan3_dsm', 'plan4_full')

    print(f"=== {args.dataset.upper()} | {args.arch} | 256² sliding ===")
    print(f"  Checkpoint: {args.ckpt}")

    # Load model
    device = 'cuda'
    if args.arch == 'plan3_rgb':
        model, ckpt = load_model_plan3_rgb(args.ckpt, device)
    elif args.arch == 'plan3_dsm':
        model, ckpt = load_model_plan3_dsm(args.ckpt, device, decode_channels=args.decode_channels)
    elif args.arch == 'plan4_full':
        model, ckpt = load_model_plan4_full(args.ckpt, device)
    else:
        raise ValueError(f"Unknown arch: {args.arch}")

    print(f"  Loaded epoch {ckpt.get('epoch', '?')}, best_v={ckpt.get('best_v', 0):.1f}%")

    # Dataset config
    if args.dataset == 'vaihingen':
        tiles = VAIHINGEN_VAL
        img_dir = '/root/autodl-tmp/dataset/Vaihingen/top'
        gt_dir = '/root/autodl-tmp/dataset/Vaihingen/gts_for_participants'
        dsm_dir = '/root/autodl-tmp/dataset/Vaihingen/dsm'
        img_suf, gt_suf = '.tif', '.tif'
        dsm_fn = lambda t: f"09cm_matching_area{t.replace('top_mosaic_09cm_area', '')}.tif"
    else:
        tiles = POTSDAM_VAL
        img_dir = '/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB'
        gt_dir = '/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants'
        dsm_dir = '/root/autodl-tmp/dataset/Potsdam/1_DSM'
        img_suf, gt_suf = '_RGB.tif', '_label.tif'
        dsm_fn = lambda t: f"potsdam_{t.replace('top_potsdam_', '')}.tif"

    results = []
    for tile in tiles:
        oa, miou, ious, oas = evaluate_tile(
            model, tile, img_dir, gt_dir, dsm_dir, img_suf, gt_suf, dsm_fn,
            use_dsm=use_dsm, arch=args.arch)
        results.append({'tile': tile, 'oa': float(oa), 'miou': float(miou),
                        **{f'{k}_iou': v for k, v in ious.items()},
                        **{f'{k}_oa': v for k, v in oas.items()}})

    # Summary
    avg_oa = np.mean([r['oa'] for r in results])
    avg_miou = np.mean([r['miou'] for r in results])
    avg_pc_iou = {c: float(np.mean([r[f'{c}_iou'] for r in results])) for c in CLASS_NAMES}
    avg_pc_oa = {c: float(np.mean([r[f'{c}_oa'] for r in results])) for c in CLASS_NAMES}

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {args.dataset.upper()} | {args.arch} | 256² sliding window")
    print(f"  Overall OA : {avg_oa:.2f}%")
    print(f"  Overall mIoU: {avg_miou:.2f}%")
    print(f"  Per-class IoU: " + ", ".join(f"{c}={avg_pc_iou[c]:.1f}" for c in CLASS_NAMES))
    print(f"  Per-class OA : " + ", ".join(f"{c}={avg_pc_oa[c]:.1f}" for c in CLASS_NAMES))
    print(f"{'=' * 70}")

    # Save
    out_dir = args.output or os.path.dirname(args.ckpt)
    out_file = f'{out_dir}/eval_256_{args.dataset}_{args.arch}.json'
    with open(out_file, 'w') as f:
        json.dump({
            'avg_oa': float(avg_oa), 'avg_miou': float(avg_miou),
            'per_class_iou': avg_pc_iou, 'per_class_oa': avg_pc_oa,
            'tiles': results
        }, f, indent=2)
    print(f"\nSaved → {out_file}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
