#!/usr/bin/env python3
"""Quick 256² eval for p3b models (LoRA / MFNet-style)."""
import os, sys, glob, numpy as np, torch, torch.nn.functional as F, json
from PIL import Image
SE = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, SE); sys.path.insert(0, '/root/Mynet/RS-SAM-p3b')
from dataset_adapter import _rgb_to_class, VAIHINGEN_VAL, POTSDAM_VAL

CKPTS = (sorted(glob.glob('/root/autodl-tmp/runs/plan3_mfnet_r*_*/best_model.pt')) +
         sorted(glob.glob('/root/autodl-tmp/runs/plan3_b_lora_r*_*/best_model.pt')))
CKPT = CKPTS[-1] if CKPTS else None  # MFNet is newer (checked first but sorted alphabetically)
OUT = os.path.dirname(CKPT) if CKPT else '/root/autodl-tmp/runs'
CLASS_NAMES = ['road','building','grass','tree','car']
is_mfnet = 'mfnet' in (CKPT or '')

print(f"Checkpoint: {CKPT}")
print(f"Model type: {'MFNet' if is_mfnet else 'LoRA'}")

_prev = os.getcwd(); os.chdir(SE)
from sam3 import build_sam3_image_model
os.chdir(_prev)
sam3 = build_sam3_image_model(bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
                              checkpoint_path=f'{SE}/weights/sam3/sam3.pt', device='cuda')

if is_mfnet:
    from lora_mfnet import LoRASAM3MFNet as ModelCls
    m = ModelCls(sam3, lora_rank=8, lora_alpha=16, num_classes=5, dropout=0.1).cuda()
else:
    from lora_sam3 import LoRASAM3UNetFormer as ModelCls
    m = ModelCls(sam3, lora_rank=8, lora_alpha=16, num_classes=5, dropout=0.1).cuda()
m.resolution = 1008
ckpt = torch.load(CKPT, map_location='cuda', weights_only=False)
m.load_state_dict(ckpt['model'], strict=False)
m.eval()
print(f"Epoch {ckpt['epoch']}, crop mIoU={ckpt['best_v']:.1f}%")

dataset = 'potsdam' if 'potsdam' in CKPT else 'vaihingen'
tiles = POTSDAM_VAL if dataset == 'potsdam' else VAIHINGEN_VAL
if dataset == 'vaihingen':
    img_dir = '/root/autodl-tmp/dataset/Vaihingen/top'
    gt_dir = '/root/autodl-tmp/dataset/Vaihingen/gts_for_participants'
    isuf, gsuf = '.tif', '.tif'
    dsm_dir = '/root/autodl-tmp/dataset/Vaihingen/dsm'
    dsm_pat = 'dsm_09cm_matching_area{}.tif'
else:
    img_dir = '/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB'
    gt_dir = '/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants'
    isuf, gsuf = '_RGB.tif', '_label.tif'
    dsm_dir = '/root/autodl-tmp/dataset/Potsdam/1_DSM'
    dsm_pat = 'dsm_potsdam_{}.tif'

def sliding_256(th, tw, img, dsm=None, stride=128):
    ps = np.zeros((th,tw), dtype=np.float64); ct = np.zeros((th,tw), dtype=np.float64)
    for y in range(0,th-128,stride):
        for x in range(0,tw-128,stride):
            y2,x2=min(y+256,th),min(x+256,tw); ph,pw=y2-y,x2-x
            if ph<128 or pw<128: continue
            pr=torch.from_numpy(img[y:y2,x:x2]).permute(2,0,1).float().unsqueeze(0)/255.0
            pr=F.interpolate(pr,(1008,1008),mode='bilinear',align_corners=False)
            with torch.no_grad():
                if dsm is not None:
                    pd=torch.from_numpy(dsm[y:y2,x:x2]).float().unsqueeze(0).unsqueeze(0)
                    pd=F.interpolate(pd,(1008,1008),mode='bilinear',align_corners=False)
                    logits=m(pr.cuda(), pd.squeeze(1).cuda())
                else:
                    logits=m(pr.cuda())
                logits=F.interpolate(logits,(256,256),mode='bilinear',align_corners=False)
                p=logits.argmax(1)[0,:ph,:pw].cpu().numpy().astype(np.float64)
            im,jm=min(16,ph//4),min(16,pw//4)
            ps[y+im:y2-im,x+jm:x2-jm]+=p[im:ph-im,jm:pw-jm]
            ct[y+im:y2-im,x+jm:x2-jm]+=1.0
    ct[ct==0]=1.0; return np.round(ps/ct).astype(np.int64)

results=[]
for tile in tiles:
    ip=f'{img_dir}/{tile}{isuf}'; gp=f'{gt_dir}/{tile}{gsuf}'
    img=np.array(Image.open(ip).convert('RGB'))
    gt=_rgb_to_class(np.array(Image.open(gp).convert('RGB')))

    # Load DSM for MFNet model
    dsm = None
    if is_mfnet:
        stem = tile.replace('top_potsdam_','') if dataset=='potsdam' else tile.replace('top_mosaic_09cm_area','')
        dp = dsm_pat.format(stem)
        dp = f'{dsm_dir}/{dp}'
        if os.path.exists(dp):
            dsm = np.array(Image.open(dp)).astype(np.float32)
            dsm = (dsm-dsm.min())/max(dsm.max()-dsm.min(),1e-8)

    print(f'{tile} ({img.shape[1]}x{img.shape[0]})...',end=' ',flush=True)
    pred=sliding_256(img.shape[0],img.shape[1],img,dsm)
    mask=gt!=255
    oa=(pred[mask]==gt[mask]).sum()/mask.sum()*100
    ious={}
    for c in range(5):
        pc,lc=pred==c,gt==c
        i=float((pc&lc).sum()); u=float((pc|lc).sum())
        ious[CLASS_NAMES[c]]=i/u*100 if u>0 else 0.0
    miou=np.mean(list(ious.values()))
    results.append({'tile':tile,'oa':float(oa),'miou':float(miou),**ious})
    print(f'OA={oa:.1f}% mIoU={miou:.1f}%')

avg_oa=np.mean([r['oa'] for r in results]); avg_miou=np.mean([r['miou'] for r in results])
print(f"\n{'='*60}")
print(f"FINAL (256): OA={avg_oa:.2f}%  mIoU={avg_miou:.2f}%")
for c in CLASS_NAMES:
    print(f"  {c}: {np.mean([r[c] for r in results]):.1f}")

tag = 'mfnet' if is_mfnet else 'lora'
json.dump({'avg_oa':float(avg_oa),'avg_miou':float(avg_miou),
           'per_class':{c:float(np.mean([r[c] for r in results])) for c in CLASS_NAMES},
           'tiles':results}, open(f'{OUT}/eval_256_{dataset}_{tag}.json','w'), indent=2)
del m,sam3; torch.cuda.empty_cache()
print('Done!')
