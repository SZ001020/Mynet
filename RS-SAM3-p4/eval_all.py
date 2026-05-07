#!/usr/bin/env python3
"""Re-evaluate all top-3 models with 256² sliding window."""
import os, sys, numpy as np, torch, torch.nn.functional as F, json
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

SE = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, SE); sys.path.insert(0, '/root/Mynet/RS-SAM3-p4')
sys.path.insert(0, '/root/Mynet/RS-SAM3-p3r'); sys.path.insert(0, '/root/Mynet/RS-SAM-p3b')
from dataset_adapter import _rgb_to_class, VAIHINGEN_VAL
CLASS_NAMES = ['road','building','grass','tree','car']

models = [
    ('VPT+MFNet Decoder (frozen best)',
     '/root/autodl-tmp/runs/plan3_mfnetdec_dsm_20260506_220419/best_model.pt',
     'p3b', 'train_mfnet_decoder', 'VPT_MFNetDecoder'),
    ('VPT+DSM UNetFormer (256² win)',
     '/root/autodl-tmp/runs/plan3_256win_dsm_20260505_171252/best_model.pt',
     'p3r', 'adapter_unet', 'AdapterSAM3UNetFormerDSM'),
    ('LoRA RGB (256² win)',
     '/root/autodl-tmp/runs/plan3_256win_rgb_20260506_005800/best_model.pt',
     'p3b', 'lora_sam3', 'LoRASAM3UNetFormer'),
]

for name, ckpt_path, src, module_name, class_name in models:
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"  Checkpoint: {ckpt_path}")

    if src == 'p3b':
        sys.path.insert(0, '/root/Mynet/RS-SAM-p3b')
    elif src == 'p3r':
        sys.path.insert(0, '/root/Mynet/RS-SAM3-p3r')

    # Import model class dynamically
    if class_name == 'VPT_MFNetDecoder':
        from train_mfnet_decoder import VPT_MFNetDecoder as ModelCls
    elif class_name == 'AdapterSAM3UNetFormerDSM':
        from adapter_unet import AdapterSAM3UNetFormerDSM as ModelCls
    else:
        from lora_sam3 import LoRASAM3UNetFormer as ModelCls

    _prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    os.chdir(_prev)
    sam3 = build_sam3_image_model(
        bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device='cuda')

    if 'DSM' in name or 'dsm' in ckpt_path:
        m = ModelCls(sam3, adapter_bottleneck=32, num_classes=5, dropout=0.1,
                     use_dsm=True).cuda()
        use_dsm = True
    else:
        m = ModelCls(sam3, lora_rank=8, lora_alpha=16, num_classes=5, dropout=0.1).cuda()
        use_dsm = False
    m.resolution = 1008
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    m.load_state_dict(ckpt['model'], strict=False)
    m.eval()
    print(f"  Epoch {ckpt['epoch']}, crop best={ckpt['best_v']:.1f}%")

    results = []
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
                        logits = m(pr.cuda(), pd.squeeze(1).cuda())
                    else:
                        logits = m(pr.cuda())
                    logits = F.interpolate(logits, (256, 256), mode='bilinear', align_corners=False)
                    p = logits.argmax(1)[0, :ph, :pw].cpu().numpy().astype(np.float64)
                im, jm = min(16, ph // 4), min(16, pw // 4)
                ps[y + im:y2 - im, x + jm:x2 - jm] += p[im:ph - im, jm:pw - jm]
                ct[y + im:y2 - im, x + jm:x2 - jm] += 1.0
        ct[ct == 0] = 1.0
        pred = np.round(ps / ct).astype(np.int64)
        mask = gt != 255
        oa = (pred[mask] == gt[mask]).sum() / mask.sum() * 100
        ious = {}
        for i, c in enumerate(CLASS_NAMES):
            pc = (pred == i); lc = (gt == i)
            inter = float((pc & lc).sum()); union = float((pc | lc).sum())
            ious[c] = inter / union * 100 if union > 0 else 0.0
        miou = np.mean(list(ious.values()))
        results.append({'tile': tile, 'oa': float(oa), 'miou': float(miou), **ious})
        print(f'  {tile}: OA={oa:.1f}% mIoU={miou:.1f}%')

    avg_oa = np.mean([r['oa'] for r in results])
    avg_miou = np.mean([r['miou'] for r in results])
    print(f"  -> OA={avg_oa:.2f}% mIoU={avg_miou:.2f}%")
    for c in CLASS_NAMES:
        print(f"     {c}: {np.mean([r[c] for r in results]):.1f}")

    json.dump({'name': name, 'avg_oa': float(avg_oa), 'avg_miou': float(avg_miou),
               'per_class': {c: float(np.mean([r[c] for r in results])) for c in CLASS_NAMES},
               'tiles': results},
              open(os.path.join(os.path.dirname(ckpt_path), 'eval_256_v2.json'), 'w'), indent=2)
    del m, sam3
    torch.cuda.empty_cache()

print("\n\nAll evaluations complete!")
