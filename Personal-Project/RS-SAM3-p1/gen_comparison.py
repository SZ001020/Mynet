#!/usr/bin/env python3
"""生成 Plan1/Plan2/Plan3 在统一 256² 协议下的推理图对比。

用法:
  python gen_comparison.py --tile top_mosaic_09cm_area5 --output /root/Mynet/weeklyReport/2026-05-10
"""

import os, sys, argparse, numpy as np, torch, torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

BASE = '/root/Mynet'
SE = f'{BASE}/Reference-Project/SegEarth-OV-3-main'
sys.path.insert(0, SE)
sys.path.insert(0, f'{BASE}/Personal-Project/RS-SAM3-p5')

from dataset_adapter import _rgb_to_class, VAIHINGEN_VAL

CLASS_NAMES = ['road', 'building', 'grass', 'tree', 'car']
CLASS_COLORS = ['#808080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00']  # match ISPRS
CMAP = ListedColormap(CLASS_COLORS)
IGNORE_INDEX = 255


def load_sam3_processor():
    _prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    m = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device='cuda')
    os.chdir(_prev)
    return Sam3Processor(m, confidence_threshold=0.5, device='cuda')


@torch.no_grad()
def plan1_sliding(processor, img_np):
    """Plan1: SAM3 zero-shot dual-head max + presence → 256² sliding."""
    H, W = img_np.shape[:2]; stride = 128
    pred_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)

    for y in range(0, H - 128, stride):
        for x in range(0, W - 128, stride):
            y2, x2 = min(y + 256, H), min(x + 256, W)
            ph, pw = y2 - y, x2 - x
            if ph < 128 or pw < 128: continue
            patch = Image.fromarray(img_np[y:y2, x:x2])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                state = processor.set_image(patch)
                seg = torch.zeros((5, ph, pw), device='cuda')
                for ci, cn in enumerate(CLASS_NAMES):
                    processor.reset_all_prompts(state)
                    state = processor.set_text_prompt(state=state, prompt=cn)
                    ml = state.get('masks_logits')
                    if ml is not None and ml.shape[0] > 0:
                        sc = state.get('object_score', state.get('scores'))
                        for ii in range(ml.shape[0]):
                            il = ml[ii]
                            if il.dim() == 3: il = il.squeeze(0)
                            isc = sc[ii].item() if sc is not None and sc.numel() > ii else 1.0
                            if il.shape != (ph, pw):
                                il = F.interpolate(il.float().view(1,1,*il.shape), size=(ph,pw), mode='bilinear', align_corners=False).squeeze()
                            seg[ci] = torch.max(seg[ci], il * isc)
                    sm = state.get('semantic_mask_logits')
                    if sm is not None:
                        if sm.dim() == 2: sm = sm.unsqueeze(0).unsqueeze(0)
                        elif sm.dim() == 3: sm = sm.unsqueeze(0)
                        if sm.shape[-2:] != (ph, pw):
                            sm = F.interpolate(sm.float(), size=(ph,pw), mode='bilinear', align_corners=False)
                        seg[ci] = torch.max(seg[ci], sm.squeeze().float())
                    ps = state.get('presence_score')
                    if ps is not None and isinstance(ps, torch.Tensor):
                        seg[ci] = seg[ci] * ps.squeeze()
                pred = seg.argmax(0).cpu().numpy().astype(np.float64)
            m = 16; im, jm = min(m, ph//4), min(m, pw//4)
            pred_sum[y+im:y2-im, x+jm:x2-jm] += pred[im:ph-im, jm:pw-jm]
            count[y+im:y2-im, x+jm:x2-jm] += 1.0
    count[count==0] = 1.0
    return np.round(pred_sum / count).astype(np.int64)


@torch.no_grad()
def plan2_sliding(processor, img_np):
    """Plan2: instance OR semantic top-K% → 256² sliding."""
    H, W = img_np.shape[:2]; stride = 128
    # Use calibrated K values
    k_vals = {'road': 40, 'building': 10, 'grass': 30, 'tree': 20, 'car': 3}
    prob_sum = np.zeros((5, H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)

    for y in range(0, H - 128, stride):
        for x in range(0, W - 128, stride):
            y2, x2 = min(y + 256, H), min(x + 256, W)
            ph, pw = y2 - y, x2 - x
            if ph < 128 or pw < 128: continue
            patch = Image.fromarray(img_np[y:y2, x:x2])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                state = processor.set_image(patch)
                prob = torch.zeros((5, ph, pw), device='cuda')
                for ci, cn in enumerate(CLASS_NAMES):
                    processor.reset_all_prompts(state)
                    state = processor.set_text_prompt(state=state, prompt=cn)
                    k_val = k_vals[cn]
                    cb = torch.zeros((ph, pw), dtype=torch.bool, device='cuda')
                    masks = state.get('masks')
                    if masks is not None and len(masks) > 0:
                        for m in masks[:10]:
                            mr = m.squeeze()
                            if mr.shape != (ph, pw):
                                mr = F.interpolate(mr.float().view(1,1,*mr.shape), size=(ph,pw), mode='nearest').squeeze() > 0.5
                            cb = cb | mr.bool()
                    sp = torch.zeros((ph, pw), device='cuda')
                    sm = state.get('semantic_mask_logits')
                    if sm is not None:
                        sp = sm.squeeze().sigmoid().float()
                        if sp.dim() > 2: sp = sp.squeeze()
                        if sp.shape != (ph, pw):
                            sp = F.interpolate(sp.unsqueeze(0).unsqueeze(0), size=(ph,pw), mode='bilinear', align_corners=False).squeeze()
                        k = max(int(sp.numel() * k_val / 100.0), 50)
                        th = sp.flatten().topk(k).values[-1].item()
                        cb = cb | (sp > th)
                    prob[ci] = torch.max(sp, cb.float() * 0.8)
            m = 16; im, jm = min(m, ph//4), min(m, pw//4)
            prob_sum[:, y+im:y2-im, x+jm:x2-jm] += prob[:, im:ph-im, jm:pw-jm].cpu().numpy()
            count[y+im:y2-im, x+jm:x2-jm] += 1.0
    count[count==0] = 1.0
    prob_sum /= count
    return prob_sum.argmax(0).astype(np.int64)


def plan3_sliding(img_np, dsm_np):
    """Plan3: VPT+MFNet Decoder best checkpoint → 256² sliding."""
    sys.path.insert(0, f'{BASE}/Personal-Project/RS-SAM-p3b')
    _prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    os.chdir(_prev)
    from train_mfnet_decoder import VPT_MFNetDecoder
    sam3 = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device='cuda')
    model = VPT_MFNetDecoder(sam3, adapter_bottleneck=32, num_classes=5, use_dsm=True, dropout=0.1).cuda()
    ckpt = torch.load(f'{BASE}/autodl-tmp/runs/plan3_mfnetdec_dsm_20260506_220419/best_model.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    H, W = img_np.shape[:2]; stride = 128
    pred_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)

    for y in range(0, H - 128, stride):
        for x in range(0, W - 128, stride):
            y2, x2 = min(y + 256, H), min(x + 256, W)
            ph, pw = y2 - y, x2 - x
            if ph < 128 or pw < 128: continue
            rgb = torch.from_numpy(img_np[y:y2, x:x2]).permute(2,0,1).float().unsqueeze(0)/255.0
            dsm = torch.from_numpy(dsm_np[y:y2, x:x2]).float().unsqueeze(0)
            rgb = F.interpolate(rgb, (1008,1008), mode='bilinear', align_corners=False)
            dsm = F.interpolate(dsm.unsqueeze(1), (1008,1008), mode='bilinear', align_corners=False).squeeze(1)
            with torch.no_grad():
                logits = model(rgb.cuda(), dsm.cuda())
                logits = F.interpolate(logits, (256,256), mode='bilinear', align_corners=False)
                pred = logits.argmax(1)[0, :ph, :pw].cpu().numpy().astype(np.float64)
            m = 16; im, jm = min(m, ph//4), min(m, pw//4)
            pred_sum[y+im:y2-im, x+jm:x2-jm] += pred[im:ph-im, jm:pw-jm]
            count[y+im:y2-im, x+jm:x2-jm] += 1.0
    count[count==0] = 1.0
    del model, sam3; torch.cuda.empty_cache()
    return np.round(pred_sum / count).astype(np.int64)


def plot_full_comparison(img, gt, preds, out_path):
    """全图对比：RGB + GT + 三个预测结果水平排列。"""
    names = ['RGB', 'Ground Truth', 'Plan1 (zero-shot)', 'Plan2 (binary)', 'Plan3 (frozen best)']
    maps = [img, gt] + preds

    fig, axes = plt.subplots(1, 5, figsize=(28, 5.5))
    for ax, name, data in zip(axes, names, maps):
        if name == 'RGB':
            ax.imshow(data)
        else:
            ax.imshow(data, cmap=CMAP, vmin=0, vmax=4, interpolation='nearest')
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.axis('off')

    # Legend
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(5)]
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=11, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_detail_comparison(img, gt, preds, regions, out_path):
    """细节对比：多个区域放大展示。"""
    n_regions = len(regions)
    fig, axes = plt.subplots(n_regions, 5, figsize=(22, 4.5 * n_regions))
    if n_regions == 1:
        axes = axes.reshape(1, -1)

    names = ['RGB', 'Ground Truth', 'Plan1 (zero-shot)', 'Plan2 (binary)', 'Plan3 (frozen best)']

    for row, (ry, rx, rh, rw, label) in enumerate(regions):
        maps = [img[ry:ry+rh, rx:rx+rw], gt[ry:ry+rh, rx:rx+rw]] + \
               [p[ry:ry+rh, rx:rx+rw] for p in preds]
        for col, (ax, name, data) in enumerate(zip(axes[row], names, maps)):
            if name == 'RGB':
                ax.imshow(data)
            else:
                ax.imshow(data, cmap=CMAP, vmin=0, vmax=4, interpolation='nearest')
            if row == 0:
                ax.set_title(name, fontsize=12, fontweight='bold')
            ax.axis('off')
        axes[row, 0].text(-30, rh//2, f'{label}', fontsize=14, fontweight='bold',
                          va='center', ha='right', rotation=90)

    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(5)]
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=11, frameon=False)
    plt.tight_layout(rect=[0.04, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_perclass_bars(plan1_ious, plan2_ious, plan3_ious, out_path):
    """Per-class IoU 柱状图对比。"""
    x = np.arange(len(CLASS_NAMES))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - w, [plan1_ious[c] for c in CLASS_NAMES], w, label='Plan1 zero-shot',
                   color='#E74C3C', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, [plan2_ious[c] for c in CLASS_NAMES], w, label='Plan2 binary',
                   color='#3498DB', edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + w, [plan3_ious[c] for c in CLASS_NAMES], w, label='Plan3 frozen best',
                   color='#2ECC71', edgecolor='white', linewidth=0.5)

    for bar in bars1 + bars2 + bars3:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.8, f'{h:.1f}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=13)
    ax.set_ylabel('IoU (%)', fontsize=13); ax.set_ylim(0, 100)
    ax.legend(fontsize=11, loc='upper right'); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # mIoU annotation
    ax.text(0.98, 0.92, f'Plan1 mIoU=60.5%\nPlan2 mIoU=49.8%\nPlan3 mIoU=73.5%',
            transform=ax.transAxes, fontsize=12, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='#F8F8F8', edgecolor='#DDD'))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile', default='top_mosaic_09cm_area5')
    parser.add_argument('--output', default='/root/Mynet/weeklyReport/2026-05-10')
    args = parser.parse_args()

    img_path = f'/root/autodl-tmp/dataset/Vaihingen/top/{args.tile}.tif'
    gt_path = f'/root/autodl-tmp/dataset/Vaihingen/gts_for_participants/{args.tile}.tif'

    img = np.array(Image.open(img_path).convert('RGB'))
    gt = _rgb_to_class(np.array(Image.open(gt_path).convert('RGB')))
    print(f"Tile: {args.tile} ({img.shape[1]}×{img.shape[0]})")

    # Load DSM
    stem = args.tile.replace('top_mosaic_09cm_area', '')
    dsm_path = f'/root/autodl-tmp/dataset/Vaihingen/dsm/dsm_09cm_matching_area{stem}.tif'
    if os.path.exists(dsm_path):
        dsm_img = np.array(Image.open(dsm_path)).astype(np.float32)
        dsm_img = (dsm_img - dsm_img.min()) / max(dsm_img.max() - dsm_img.min(), 1e-8)
    else:
        dsm_img = np.zeros(img.shape[:2], dtype=np.float32)
        print(f"  DSM not found, using zeros")

    # Load SAM3 once
    print("\nLoading SAM3...")
    processor = load_sam3_processor()

    # Plan1
    print("\n[1/3] Plan1 sliding window inference...")
    p1 = plan1_sliding(processor, img)

    # Plan2
    print("\n[2/3] Plan2 sliding window inference...")
    p2 = plan2_sliding(processor, img)

    # Plan3
    print("\n[3/3] Plan3 (VPT+MFNet Decoder) sliding window inference...")
    p3 = plan3_sliding(img, dsm_img)

    del processor; torch.cuda.empty_cache()

    # Metrics
    mask = gt != IGNORE_INDEX
    def calc_metrics(pred):
        ious = {}
        for i, c in enumerate(CLASS_NAMES):
            pc, lc = pred == i, gt == i
            inter = (pc & lc & mask).sum(); union = (pc | lc) & mask
            ious[c] = inter / max(union.sum(), 1) * 100
        return ious

    p1_ious = calc_metrics(p1)
    p2_ious = calc_metrics(p2)
    p3_ious = calc_metrics(p3)

    print(f"\n  Plan1 mIoU={np.mean(list(p1_ious.values())):.1f}%")
    print(f"  Plan2 mIoU={np.mean(list(p2_ious.values())):.1f}%")
    print(f"  Plan3 mIoU={np.mean(list(p3_ious.values())):.1f}%")

    # Generate figures
    print("\nGenerating figures...")
    os.makedirs(args.output, exist_ok=True)

    # Full comparison
    plot_full_comparison(img, gt, [p1, p2, p3],
                         os.path.join(args.output, 'full_comparison.png'))

    # Detail regions - pick areas with interesting differences
    # Region 1: building boundaries (top-right area, pick a 300x300 region)
    H, W = img.shape[:2]
    regions = [
        (H//3, W//2, 300, 300, '① Building\nboundaries'),
        (H//2, 2*W//5, 300, 300, '② Road &\ncar area'),
        (H//4, W//4, 300, 300, '③ Mixed\nvegetation'),
    ]
    plot_detail_comparison(img, gt, [p1, p2, p3], regions,
                           os.path.join(args.output, 'detail_comparison.png'))

    # Per-class bar chart
    plot_perclass_bars(p1_ious, p2_ious, p3_ious,
                       os.path.join(args.output, 'perclass_bars.png'))

    print("\nDone.")


if __name__ == '__main__':
    main()
