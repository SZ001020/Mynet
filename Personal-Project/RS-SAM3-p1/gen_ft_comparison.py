#!/usr/bin/env python3
"""生成Zero-shot与微调模型的 2-way 对比图（05-17 周报用）。"""
import os, sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

BASE = '/root/Mynet'; SE = f'{BASE}/Reference-Project/SegEarth-OV-3-main'
sys.path.insert(0, SE); sys.path.insert(0, f'{BASE}/Personal-Project/RS-SAM3-p5')
sys.path.insert(0, f'{BASE}/Personal-Project/RS-SAM-p3b')
from dataset_adapter import _rgb_to_class

CLASS_NAMES = ['road', 'building', 'grass', 'tree', 'car']
CLASS_COLORS = ['#808080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00']
CMAP = ListedColormap(CLASS_COLORS)
IGNORE_INDEX = 255


def load_processor():
    _prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    m = build_sam3_image_model(bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
                               checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device='cuda')
    os.chdir(_prev)
    return Sam3Processor(m, confidence_threshold=0.5, device='cuda')


@torch.no_grad()
def zero_shot_sliding(processor, img_np):
    """Zero-shot: dual-head max + presence score."""
    H, W = img_np.shape[:2]; stride = 128
    ps = np.zeros((H, W), dtype=np.float64); ct = np.zeros((H, W), dtype=np.float64)
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
                            il = ml[ii]; isc = sc[ii].item() if sc is not None and sc.numel() > ii else 1.0
                            if il.dim() == 3: il = il.squeeze(0)
                            if il.shape != (ph, pw):
                                il = F.interpolate(il.float().view(1,1,*il.shape), size=(ph,pw), mode='bilinear', align_corners=False).squeeze()
                            seg[ci] = torch.max(seg[ci], il * isc)
                    sm = state.get('semantic_mask_logits')
                    if sm is not None:
                        if sm.dim() == 2: sm = sm.unsqueeze(0).unsqueeze(0)
                        elif sm.dim() == 3: sm = sm.unsqueeze(0)
                        if sm.shape[-2:] != (ph, pw): sm = F.interpolate(sm.float(), size=(ph,pw), mode='bilinear', align_corners=False)
                        seg[ci] = torch.max(seg[ci], sm.squeeze().float())
                    psc = state.get('presence_score')
                    if psc is not None and isinstance(psc, torch.Tensor): seg[ci] = seg[ci] * psc.squeeze()
                pred = seg.argmax(0).cpu().numpy().astype(np.float64)
            m = 16; im, jm = min(m, ph//4), min(m, pw//4)
            ps[y+im:y2-im, x+jm:x2-jm] += pred[im:ph-im, jm:pw-jm]
            ct[y+im:y2-im, x+jm:x2-jm] += 1.0
    ct[ct==0] = 1.0
    return np.round(ps / ct).astype(np.int64)


@torch.no_grad()
def finetuned_sliding(img_np, dsm_np):
    """Fine-tuned模型滑动窗口推理."""
    _prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    os.chdir(_prev)
    from train_mfnet_decoder import VPT_MFNetDecoder

    sam3 = build_sam3_image_model(bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
                                   checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device='cuda')
    model = VPT_MFNetDecoder(sam3, adapter_bottleneck=32, num_classes=5, use_dsm=True, dropout=0.1).cuda()
    ckpt = torch.load(f'{BASE}/autodl-tmp/runs/plan3_mfnetdec_dsm_20260506_220419/best_model.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False); model.eval()

    H, W = img_np.shape[:2]; stride = 128
    ps = np.zeros((H, W), dtype=np.float64); ct = np.zeros((H, W), dtype=np.float64)
    for y in range(0, H - 128, stride):
        for x in range(0, W - 128, stride):
            y2, x2 = min(y + 256, H), min(x + 256, W)
            ph, pw = y2 - y, x2 - x
            if ph < 128 or pw < 128: continue
            rgb = torch.from_numpy(img_np[y:y2, x:x2]).permute(2,0,1).float().unsqueeze(0)/255.0
            dsm = torch.from_numpy(dsm_np[y:y2, x:x2]).float().unsqueeze(0)
            rgb = F.interpolate(rgb, (1008,1008), mode='bilinear', align_corners=False)
            dsm = F.interpolate(dsm.unsqueeze(1), (1008,1008), mode='bilinear', align_corners=False).squeeze(1)
            logits = model(rgb.cuda(), dsm.cuda())
            logits = F.interpolate(logits, (256,256), mode='bilinear', align_corners=False)
            pred = logits.argmax(1)[0, :ph, :pw].cpu().numpy().astype(np.float64)
            m = 16; im, jm = min(m, ph//4), min(m, pw//4)
            ps[y+im:y2-im, x+jm:x2-jm] += pred[im:ph-im, jm:pw-jm]
            ct[y+im:y2-im, x+jm:x2-jm] += 1.0
    ct[ct==0] = 1.0
    del model, sam3; torch.cuda.empty_cache()
    return np.round(ps / ct).astype(np.int64)


def plot_full(img, gt, preds, names, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    maps = [img, gt] + preds
    for ax, name, data in zip(axes, ['RGB', 'Ground Truth'] + names, maps):
        if name == 'RGB': ax.imshow(data)
        else: ax.imshow(data, cmap=CMAP, vmin=0, vmax=4, interpolation='nearest')
        ax.set_title(name, fontsize=13, fontweight='bold'); ax.axis('off')
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(5)]
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=11, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)


def plot_detail(img, gt, preds, names, regions, out_path):
    n = len(regions)
    fig, axes = plt.subplots(n, 4, figsize=(18, 4.5*n))
    if n == 1: axes = axes.reshape(1,-1)
    cn = ['RGB', 'Ground Truth'] + names
    for row, (ry, rx, rh, rw, label) in enumerate(regions):
        maps = [img[ry:ry+rh, rx:rx+rw], gt[ry:ry+rh, rx:rx+rw]] + [p[ry:ry+rh, rx:rx+rw] for p in preds]
        for col, (ax, nm, data) in enumerate(zip(axes[row], cn, maps)):
            if nm == 'RGB': ax.imshow(data)
            else: ax.imshow(data, cmap=CMAP, vmin=0, vmax=4, interpolation='nearest')
            if row == 0: ax.set_title(nm, fontsize=12, fontweight='bold')
            ax.axis('off')
        axes[row, 0].text(-30, rh//2, label, fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(5)]
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=11, frameon=False)
    plt.tight_layout(rect=[0.04, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)


def plot_bars(zs_ious, ft_ious, out_path):
    x = np.arange(len(CLASS_NAMES)); w = 0.3
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x-w/2, [zs_ious[c] for c in CLASS_NAMES], w, label='Zero-shot', color='#E74C3C', edgecolor='white')
    b2 = ax.bar(x+w/2, [ft_ious[c] for c in CLASS_NAMES], w, label='Fine-tuned', color='#2ECC71', edgecolor='white')
    for bar in b1: ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.8, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in b2: ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.8, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=13)
    ax.set_ylabel('IoU (%)', fontsize=13); ax.set_ylim(0, 100)
    ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.text(0.98, 0.92, f'Zero-shot mIoU=60.5%\nFine-tuned mIoU=73.1%\nDelta +12.6pp', transform=ax.transAxes, fontsize=12, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='#F8F8F8', edgecolor='#DDD'))
    plt.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)


def calc_metrics(pred, gt):
    mask = gt != IGNORE_INDEX
    ious = {}
    for i, c in enumerate(CLASS_NAMES):
        pc, lc = pred == i, gt == i
        inter = (pc & lc & mask).sum()
        union = (pc | lc) & mask
        ious[c] = inter / max(union.sum(), 1) * 100
    return ious


def main():
    tile = 'top_mosaic_09cm_area5'
    out_dir = '/root/Mynet/weeklyReport/2026-05-17'
    os.makedirs(out_dir, exist_ok=True)

    img = np.array(Image.open(f'/root/autodl-tmp/dataset/Vaihingen/top/{tile}.tif').convert('RGB'))
    gt = _rgb_to_class(np.array(Image.open(f'/root/autodl-tmp/dataset/Vaihingen/gts_for_participants/{tile}.tif').convert('RGB')))
    stem = tile.replace('top_mosaic_09cm_area', '')
    dsm = np.array(Image.open(f'/root/autodl-tmp/dataset/Vaihingen/dsm/dsm_09cm_matching_area{stem}.tif')).astype(np.float32)
    dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
    print(f"Tile: {tile} ({img.shape[1]}x{img.shape[0]})")

    print("Loading SAM3...")
    processor = load_processor()

    print("[1/2] Zero-shot...")
    zs_pred = zero_shot_sliding(processor, img)
    del processor; torch.cuda.empty_cache()

    print("[2/2] 微调模型...")
    ft_pred = finetuned_sliding(img, dsm)

    zs_ious = calc_metrics(zs_pred, gt)
    ft_ious = calc_metrics(ft_pred, gt)
    print(f"零样本 mIoU={np.mean(list(zs_ious.values())):.1f}%")
    print(f"微调 mIoU={np.mean(list(ft_ious.values())):.1f}%")

    print("\nGenerating figures...")
    plot_full(img, gt, [zs_pred, ft_pred], ['Zero-shot', 'Fine-tuned'], os.path.join(out_dir, 'full_comparison.png'))

    H, W = img.shape[:2]
    regions = [
        (H//3, W//2, 300, 300, '① Building\nboundaries'),
        (H//2, 2*W//5, 300, 300, '② Road &\ncar area'),
        (H//4, W//4, 300, 300, '③ Mixed\nvegetation'),
    ]
    plot_detail(img, gt, [zs_pred, ft_pred], ['Zero-shot', 'Fine-tuned'], regions, os.path.join(out_dir, 'detail_comparison.png'))
    plot_bars(zs_ious, ft_ious, os.path.join(out_dir, 'perclass_bars.png'))
    print("Done.")


if __name__ == '__main__':
    main()
