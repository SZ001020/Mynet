#!/usr/bin/env python3
"""通用评估脚本：基于 model_registry.py 自动加载任意 checkpoint 并运行 256² 评估。

用法:
  python eval_universal.py                    # 评估所有注册模型
  python eval_universal.py --name "VPT+MFNet" # 评估指定模型（模糊匹配）
  python eval_universal.py --list             # 列出所有注册模型
"""

import os, sys, argparse, numpy as np, torch, torch.nn.functional as F, json, importlib
from PIL import Image
import sys; sys.path.insert(0, '/root/Mynet')
from model_registry import REGISTRY, MFNET_BASELINES

BASE = '/root/Mynet'
SE = f'{BASE}/Reference-Project/SegEarth-OV-3-main'
sys.path.insert(0, SE)
CLASS_NAMES = ['road', 'building', 'grass', 'tree', 'car']


def load_model(entry):
    """Load model from registry entry."""
    src_dir = f'{BASE}/{entry["source"]}'
    sys.path.insert(0, src_dir)

    # Import module and class
    mod = importlib.import_module(entry["module"])
    ModelCls = getattr(mod, entry["class"])

    # Load SAM3
    _prev = os.getcwd()
    os.chdir(SE)
    from sam3 import build_sam3_image_model
    os.chdir(_prev)
    sam3 = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device='cuda')

    model = ModelCls(sam3, **entry["kwargs"]).cuda()
    model.resolution = 1008
    ckpt = torch.load(entry["ckpt"], map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model, ckpt, entry["use_dsm"]


def sliding_eval(model, use_dsm):
    """256² sliding window evaluation on Vaihingen test tiles."""
    from dataset_adapter import _rgb_to_class, VAIHINGEN_VAL

    per_class_correct = {c: 0.0 for c in CLASS_NAMES}  # TP
    per_class_total = {c: 0.0 for c in CLASS_NAMES}   # TP + FN (GT pixels)
    per_class_pred = {c: 0.0 for c in CLASS_NAMES}    # TP + FP (pred pixels)
    per_class_inter = {c: 0.0 for c in CLASS_NAMES}
    per_class_union = {c: 0.0 for c in CLASS_NAMES}
    total_correct = 0.0
    total_pixels = 0.0

    for tile in VAIHINGEN_VAL:
        ip = f'/root/autodl-tmp/dataset/Vaihingen/top/{tile}.tif'
        gp = f'/root/autodl-tmp/dataset/Vaihingen/gts_for_participants/{tile}.tif'
        stem = tile.replace('top_mosaic_09cm_area', '')
        dp = f'/root/autodl-tmp/dataset/Vaihingen/dsm/dsm_09cm_matching_area{stem}.tif'

        img = np.array(Image.open(ip).convert('RGB'))
        gt = _rgb_to_class(np.array(Image.open(gp).convert('RGB')))
        d = None
        if use_dsm and os.path.exists(dp):
            d = np.array(Image.open(dp)).astype(np.float32)
            d = (d - d.min()) / max(d.max() - d.min(), 1e-8)

        ps = np.zeros(img.shape[:2], dtype=np.float64)
        ct = np.zeros(img.shape[:2], dtype=np.float64)
        for y in range(0, img.shape[0] - 128, 128):
            for x in range(0, img.shape[1] - 128, 128):
                y2, x2 = min(y + 256, img.shape[0]), min(x + 256, img.shape[1])
                ph, pw = y2 - y, x2 - x
                if ph < 128 or pw < 128: continue
                pr = torch.from_numpy(img[y:y2, x:x2]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                pr = F.interpolate(pr, (1008, 1008), mode='bilinear', align_corners=False)
                with torch.no_grad():
                    if use_dsm and d is not None:
                        pd = torch.from_numpy(d[y:y2, x:x2]).float().unsqueeze(0)
                        pd = F.interpolate(pd.unsqueeze(1), (1008, 1008), mode='bilinear', align_corners=False)
                        logits = model(pr.cuda(), pd.squeeze(1).cuda())
                    else:
                        logits = model(pr.cuda())
                    logits = F.interpolate(logits, (256, 256), mode='bilinear', align_corners=False)
                    p = logits.argmax(1)[0, :ph, :pw].cpu().numpy().astype(np.float64)
                im, jm = min(16, ph // 4), min(16, pw // 4)
                ps[y + im:y2 - im, x + jm:x2 - jm] += p[im:ph - im, jm:pw - jm]
                ct[y + im:y2 - im, x + jm:x2 - jm] += 1.0
        ct[ct == 0] = 1.0
        pred = np.round(ps / ct).astype(np.int64)

        mask = gt != 255
        total_correct += (pred[mask] == gt[mask]).sum()
        total_pixels += mask.sum()
        for i, c in enumerate(CLASS_NAMES):
            pc = (pred == i); lc = (gt == i)
            per_class_correct[c] += (pc & lc).sum()  # TP
            per_class_total[c] += lc.sum()            # GT pixels (TP+FN)
            per_class_pred[c] += pc.sum()             # pred pixels (TP+FP)
            per_class_inter[c] += (pc & lc).sum()
            per_class_union[c] += (pc | lc).sum()
        print(f'  {tile} done')

    results = {}
    for c in CLASS_NAMES:
        tp = per_class_correct[c]
        gt_total = max(per_class_total[c], 1)
        pred_total = max(per_class_pred[c], 1)
        # Precision = TP / (TP + FP) = TP / pred_total
        precision = tp / pred_total * 100
        # Recall = TP / (TP + FN) = TP / gt_total (= per-class OA)
        recall = tp / gt_total * 100
        # F1 = 2 * P * R / (P + R)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        # IoU
        iou = per_class_inter[c] / max(per_class_union[c], 1) * 100

        results[c] = {
            'oa': recall,  # per-class OA = recall
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
        }

    return {
        'oa': total_correct / total_pixels * 100,
        'miou': np.mean([results[c]['iou'] for c in CLASS_NAMES]),
        'mf1': np.mean([results[c]['f1'] for c in CLASS_NAMES]),
        'mrecall': np.mean([results[c]['recall'] for c in CLASS_NAMES]),
        'per_class': results,
        'per_class_iou': {c: results[c]['iou'] for c in CLASS_NAMES},
        'per_class_recall': {c: results[c]['recall'] for c in CLASS_NAMES},
        'per_class_precision': {c: results[c]['precision'] for c in CLASS_NAMES},
        'per_class_f1': {c: results[c]['f1'] for c in CLASS_NAMES},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None, help='Model name filter (fuzzy match)')
    parser.add_argument('--list', action='store_true', help='List registered models')
    args = parser.parse_args()

    if args.list:
        for i, e in enumerate(REGISTRY):
            print(f"[{i}] {e['name']}")
            print(f"    ckpt: {e['ckpt']}")
            print(f"    source: {e['source']}/{e['module']}.py → {e['class']}")
            print()
        return

    # Filter models
    entries = REGISTRY
    if args.name:
        entries = [e for e in REGISTRY if args.name.lower() in e['name'].lower()]
        if not entries:
            print(f"No models matching '{args.name}'")
            return

    results = {}
    for entry in entries:
        print(f"\n{'='*60}")
        print(f"Evaluating: {entry['name']}")
        print(f"  Source: {entry['source']}/{entry['module']}.py → {entry['class']}")

        model, ckpt, use_dsm = load_model(entry)
        print(f"  Epoch {ckpt['epoch']}, crop best={ckpt['best_v']:.1f}%")

        # Model size
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        metrics = sliding_eval(model, use_dsm)
        metrics['name'] = entry['name']
        metrics['epoch'] = ckpt['epoch']
        metrics['params_total'] = n_params
        metrics['params_trainable'] = n_trainable
        results[entry['name']] = metrics

        print(f"  Params: {n_trainable/1e6:.1f}M trainable / {n_params/1e6:.1f}M total")
        print(f"  OA={metrics['oa']:.2f}%  mIoU={metrics['miou']:.2f}%  mF1={metrics['mf1']:.2f}%")
        for c in CLASS_NAMES:
            pc = metrics['per_class'][c]
            print(f"    {c}: OA={pc['oa']:.1f}%  F1={pc['f1']:.1f}%  IoU={pc['iou']:.1f}%  "
                  f"P={pc['precision']:.1f}%  R={pc['recall']:.1f}%")

        del model
        torch.cuda.empty_cache()

    # Save all results
    out = '/root/autodl-tmp/runs/eval_summary.json'
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
