#!/usr/bin/env python3
"""Quick 256² eval for p3r DSM model on Vaihingen."""
import os, sys, numpy as np, torch, torch.nn.functional as F, json
from PIL import Image
SE = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, SE); sys.path.insert(0, '/root/Mynet/RS-SAM3-p3r')
from adapter_unet import AdapterSAM3UNetFormerDSM
from dataset_adapter import _rgb_to_class, VAIHINGEN_VAL

import glob
CKPTS = sorted(glob.glob('/root/autodl-tmp/runs/plan3_p3r_dsm_*/best_model.pt'))
CKPT = CKPTS[-1] if CKPTS else '/root/autodl-tmp/runs/plan3_p3r_dsm_20260502_195509/best_model.pt'
OUT = os.path.dirname(CKPT)
print(f"Checkpoint: {CKPT}")
CLASS_NAMES = ['road','building','grass','tree','car']

_prev = os.getcwd(); os.chdir(SE)
from sam3 import build_sam3_image_model
os.chdir(_prev)

sam3 = build_sam3_image_model(bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
                              checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device='cuda')
m = AdapterSAM3UNetFormerDSM(sam3, adapter_bottleneck=32, num_classes=5, dropout=0.1).cuda()
m.resolution = 1008
ckpt = torch.load(CKPT, map_location='cuda', weights_only=False)
m.load_state_dict(ckpt['model'], strict=False)
m.eval()
print(f"Epoch {ckpt['epoch']}, crop mIoU={ckpt['best_v']:.1f}%")

def sliding_256(th, tw, img, dsm, stride=128):
    ps = np.zeros((th,tw), dtype=np.float64)
    ct = np.zeros((th,tw), dtype=np.float64)
    for y in range(0,th-128,stride):
        for x in range(0,tw-128,stride):
            y2,x2 = min(y+256,th), min(x+256,tw)
            ph,pw = y2-y, x2-x
            if ph<128 or pw<128: continue
            pr = torch.from_numpy(img[y:y2,x:x2]).permute(2,0,1).float().unsqueeze(0)/255.0
            pd = torch.from_numpy(dsm[y:y2,x:x2]).float().unsqueeze(0)
            pr = F.interpolate(pr,(1008,1008),mode='bilinear',align_corners=False)
            pd = F.interpolate(pd.unsqueeze(1),(1008,1008),mode='bilinear',align_corners=False)
            with torch.no_grad():
                logits = m(pr.cuda(),pd.cuda())
                logits = F.interpolate(logits,(256,256),mode='bilinear',align_corners=False)
                p = logits.argmax(1)[0,:ph,:pw].cpu().numpy().astype(np.float64)
            im,jm = min(16,ph//4), min(16,pw//4)
            ps[y+im:y2-im,x+jm:x2-jm] += p[im:ph-im,jm:pw-jm]
            ct[y+im:y2-im,x+jm:x2-jm] += 1.0
    ct[ct==0] = 1.0
    return np.round(ps/ct).astype(np.int64)

results = []
tiles = VAIHINGEN_VAL
for tile in tiles:
    ip = f'/root/autodl-tmp/dataset/Vaihingen/top/{tile}.tif'
    gp = f'/root/autodl-tmp/dataset/Vaihingen/gts_for_participants/{tile}.tif'
    stem = tile.replace('top_mosaic_09cm_area','')
    dp = f'/root/autodl-tmp/dataset/Vaihingen/dsm/dsm_09cm_matching_area{stem}.tif'
    img = np.array(Image.open(ip).convert('RGB'))
    gt = _rgb_to_class(np.array(Image.open(gp).convert('RGB')))
    d = np.array(Image.open(dp)).astype(np.float32)
    d = (d-d.min())/max(d.max()-d.min(),1e-8)
    print(f'{tile} ({img.shape[1]}x{img.shape[0]})...',end=' ',flush=True)
    pred = sliding_256(img.shape[0],img.shape[1],img,d)
    mask = gt!=255
    total = float(mask.sum())
    oa = (pred[mask]==gt[mask]).sum()/total*100
    ious = {}
    oas = {}
    for c in range(5):
        pc,lc = pred==c, gt==c
        inter = float((pc&lc).sum()); union = float((pc|lc).sum())
        ious[CLASS_NAMES[c]] = inter/union*100 if union>0 else 0.0
        tn = float(((~pc)&(~lc)&mask).sum())
        oas[CLASS_NAMES[c]] = (inter+tn)/max(total,1)*100
    miou = np.mean(list(ious.values()))
    results.append({'tile':tile,'oa':float(oa),'miou':float(miou),**ious,**{f'{k}_oa':v for k,v in oas.items()}})
    print(f'OA={oa:.1f}% mIoU={miou:.1f}%')
    print(f'  per-class IoU: {[f"{ious[c]:.1f}" for c in CLASS_NAMES]}')
    print(f'  per-class OA : {[f"{oas[c]:.1f}" for c in CLASS_NAMES]}')

avg_oa = np.mean([r['oa'] for r in results])
avg_miou = np.mean([r['miou'] for r in results])
print(f"\n{'='*60}")
print(f"FINAL (256 protocol): OA={avg_oa:.2f}%  mIoU={avg_miou:.2f}%")
print(f"  Per-class IoU: " + ", ".join(f"{c}={np.mean([r[c] for r in results]):.1f}" for c in CLASS_NAMES))
print(f"  Per-class OA : " + ", ".join(f"{c}={np.mean([r[c+'_oa'] for r in results]):.1f}" for c in CLASS_NAMES))
print(f"{'='*60}")

with open(f'{OUT}/eval_256_vaihingen_dsm.json','w') as f:
    json.dump({'avg_oa':float(avg_oa),'avg_miou':float(avg_miou),
               'per_class_iou':{c:float(np.mean([r[c] for r in results])) for c in CLASS_NAMES},
               'per_class_oa':{c:float(np.mean([r[c+'_oa'] for r in results])) for c in CLASS_NAMES},
               'tiles':results},f,indent=2)
del m, sam3; torch.cuda.empty_cache()
print('Done!')
