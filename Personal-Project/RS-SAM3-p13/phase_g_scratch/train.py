#!/usr/bin/env python3
"""P13-G from-scratch: joint train base + texture + nDSM roughness from scratch."""

from __future__ import annotations

import argparse, json, os, random, sys, time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import grey_opening

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
P7_DIR = f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt"
P11B_DIR = f"{BASE}/Personal-Project/RS-SAM3-p11/phase_b_ndsm"
P13GS_DIR = f"{BASE}/Personal-Project/RS-SAM3-p13/phase_g_scratch"
sys.path.insert(0, SE); sys.path.insert(0, P7_DIR); sys.path.insert(0, P11B_DIR); sys.path.insert(0, P13GS_DIR)

from dataset_adapter import (IGNORE_INDEX, NUM_CLASSES,
    POTSDAM_TRAIN, POTSDAM_VAL, VAIHINGEN_TRAIN, VAIHINGEN_VAL, _rgb_to_class)
from dataset_online import OnlineCropDataset, compute_ndsm
from model import Plan13GScratch
from structure_loss import structure_loss

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]


class Window256DatasetDSM(torch.utils.data.Dataset):
    def __init__(self, img_dir, gt_dir, tiles, img_suffix, gt_suffix, dsm_paths, stride=128):
        self.samples = []
        for tile in tiles:
            ip, gp, dp = f"{img_dir}/{tile}{img_suffix}", f"{gt_dir}/{tile}{gt_suffix}", dsm_paths.get(tile)
            if not os.path.exists(ip) or not os.path.exists(gp): continue
            img = np.array(Image.open(ip).convert("RGB"))
            gt = _rgb_to_class(np.array(Image.open(gp).convert("RGB")))
            dsm = compute_ndsm(np.array(Image.open(dp)).astype(np.float32)) if dp and os.path.exists(dp) else np.zeros(img.shape[:2], dtype=np.float32)
            h, w = img.shape[:2]
            for y in range(0, h - 128, stride):
                for x in range(0, w - 128, stride):
                    y2, x2 = min(y+256,h), min(x+256,w)
                    ph, pw = y2-y, x2-x
                    if ph<128 or pw<128: continue
                    patch, label, dsp = img[y:y2,x:x2], gt[y:y2,x:x2], dsm[y:y2,x:x2]
                    if ph<256 or pw<256:
                        patch=np.pad(patch,((0,256-ph),(0,256-pw),(0,0)),mode="reflect")
                        label=np.pad(label,((0,256-ph),(0,256-pw)),mode="constant",constant_values=IGNORE_INDEX)
                        dsp=np.pad(dsp,((0,256-ph),(0,256-pw)),mode="reflect")
                    if (label==IGNORE_INDEX).mean()<=0.5: self.samples.append((patch,dsp,label))
        print(f"  {len(self.samples)} windows (val)")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        i,d,l=self.samples[idx]
        return torch.from_numpy(i.copy()).permute(2,0,1).float()/255.0, torch.from_numpy(d.copy()).float(), torch.from_numpy(l.copy()).long()


def dataset_paths(dataset: str):
    if dataset=="vaihingen":
        tr,va=VAIHINGEN_TRAIN,VAIHINGEN_VAL
        return tr,va,"/root/autodl-tmp/dataset/Vaihingen/top","/root/autodl-tmp/dataset/Vaihingen/gts_for_participants",".tif",".tif",\
               {t:f'/root/autodl-tmp/dataset/Vaihingen/dsm/dsm_09cm_matching_area{t.replace("top_mosaic_09cm_area","")}.tif' for t in tr+va}
    else:
        tr,va=POTSDAM_TRAIN,POTSDAM_VAL
        return tr,va,"/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB","/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants","_RGB.tif","_label.tif",\
               {t:f'/root/autodl-tmp/dataset/Potsdam/1_DSM/dsm_potsdam_{t.replace("top_potsdam_","")}.tif' for t in tr+va}


def load_sam3(device="cuda"):
    prev=os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    m=build_sam3_image_model(bpe_path=f"{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path=f"{SE}/weights/sam3/sam3.pt",device=device)
    os.chdir(prev); return m.cuda()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    inter=torch.zeros(NUM_CLASSES,device=device); union=torch.zeros(NUM_CLASSES,device=device)
    correct=0; total=0
    for images, dsm, labels in loader:
        images,dsm,labels=images.to(device),dsm.to(device),labels.to(device)
        logits=model(images,dsm)
        logits=F.interpolate(logits,labels.shape[-2:],mode="bilinear",align_corners=False)
        pred=logits.argmax(1); mask=labels!=IGNORE_INDEX
        correct+=(pred[mask]==labels[mask]).sum().item(); total+=mask.sum().item()
        for c in range(NUM_CLASSES):
            pc,lc=pred==c,labels==c
            inter[c]+=(pc&lc).sum(); union[c]+=(pc|lc).sum()
    pciou={CLASS_NAMES[c]:(inter[c]/union[c].clamp(min=1)*100).item() for c in range(NUM_CLASSES)}
    return {"avg_oa":correct/max(total,1)*100,"avg_miou":float(np.mean(list(pciou.values()))),
            "per_class_iou":pciou}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",default="vaihingen",choices=["vaihingen","potsdam"])
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--epoch-steps",type=int,default=1000)
    p.add_argument("--lr",type=float,default=5e-5)
    p.add_argument("--dsm-lr",type=float,default=2.5e-5)
    p.add_argument("--prompt-lr",type=float,default=2.5e-5)
    p.add_argument("--adapter-bottleneck",type=int,default=32)
    p.add_argument("--resolution",type=int,default=1008)
    p.add_argument("--dsm-attn-mode",default="full",choices=["adapter","full"])
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--val-every",type=int,default=1)
    p.add_argument("--output",default="/root/autodl-tmp/runs")
    args=p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device=torch.device("cuda")
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir=os.path.join(args.output,f"plan13_g_scratch_{args.dataset}_{ts}")
    os.makedirs(out_dir,exist_ok=True)
    json.dump(vars(args),open(os.path.join(out_dir,"config.json"),"w"),indent=2,default=str)

    print(f"P13-G from-scratch: base + texture + nDSM roughness (joint, seed={args.seed})")
    print(f"  Dataset: {args.dataset}  Output: {out_dir}")

    tr,va,idir,gdir,isf,gsf,dsp=dataset_paths(args.dataset)
    tds=OnlineCropDataset(idir,gdir,tr,isf,gsf,dsp,is_train=True,crop_size=256,epoch_steps=args.epoch_steps,batch_size=args.batch)
    vds=Window256DatasetDSM(idir,gdir,va,isf,gsf,dsp)
    tl=torch.utils.data.DataLoader(tds,batch_size=args.batch,shuffle=False,num_workers=0,pin_memory=True)
    vl=torch.utils.data.DataLoader(vds,batch_size=1,shuffle=False,num_workers=0)

    print("\nBuilding model (from scratch)...")
    sam3=load_sam3()
    model=Plan13GScratch(sam3,adapter_bottleneck=args.adapter_bottleneck,num_classes=NUM_CLASSES,dropout=0.1,
        dsm_attn_mode=args.dsm_attn_mode,checkpoint_attn=False,resolution=args.resolution).cuda()
    model.train()

    prompt_p=list(model.prompt_encoder.parameters()); prompt_ids={id(p) for p in prompt_p}
    dsm_p=list(model.dsm_encoder.parameters()); dsm_ids={id(p) for p in dsm_p}
    other_p=[p for p in model.parameters() if p.requires_grad and id(p) not in prompt_ids and id(p) not in dsm_ids]
    print(f"  other={sum(p.numel() for p in other_p):,} dsm={sum(p.numel() for p in dsm_p):,} prompt={sum(p.numel() for p in prompt_p):,}")

    opt=torch.optim.AdamW([{"params":other_p,"lr":args.lr},{"params":dsm_p,"lr":args.dsm_lr},{"params":prompt_p,"lr":args.prompt_lr}],weight_decay=1e-3)
    sch=torch.optim.lr_scheduler.MultiStepLR(opt,[25,35,45],gamma=0.1)
    scaler=torch.amp.GradScaler("cuda")

    hist={"loss":[],"metrics":[],"lr":[]}
    best_miou,best_metrics=0.0,None
    for epoch in range(1,args.epochs+1):
        model.train(); start=time.time(); running=0.0
        for step,(images,dsm,labels) in enumerate(tl):
            images,dsm,labels=images.to(device),dsm.to(device),labels.to(device)
            with torch.amp.autocast("cuda",dtype=torch.bfloat16):
                logits=model(images,dsm)
                logits=F.interpolate(logits,labels.shape[-2:],mode="bilinear",align_corners=False)
                loss=structure_loss(logits.float(),labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update()
            running+=loss.item()
            if step%100==0: print(f"  E{epoch:03d}/{args.epochs} B{step:04d}/{len(tl)} loss={loss.item():.4f} lr={sch.get_last_lr()[0]:.2e}")
        sch.step()
        avg=running/max(len(tl),1)
        metrics=None
        if epoch==1 or epoch%args.val_every==0 or epoch==args.epochs:
            metrics=validate(model,vl,device)
            print(f"  E{epoch:03d}: loss={avg:.4f} OA={metrics['avg_oa']:.2f}% mIoU={metrics['avg_miou']:.2f}% best={best_miou:.2f}% time={time.time()-start:.0f}s")
            print(f"    IoU: {metrics['per_class_iou']}")
            if metrics["avg_miou"]>best_miou:
                best_miou=metrics["avg_miou"]; best_metrics=metrics
                torch.save({"epoch":epoch,"model":model.state_dict(),"best_v":best_miou,"metrics":metrics,"args":vars(args)},os.path.join(out_dir,"best_model.pt"))
        hist["loss"].append(avg); hist["lr"].append(sch.get_last_lr()); hist["metrics"].append(metrics)
        json.dump(hist,open(os.path.join(out_dir,"history.json"),"w"),indent=2)
        if best_metrics is not None: json.dump(best_metrics,open(os.path.join(out_dir,"metrics.json"),"w"),indent=2)
    print(f"\nDone. Best mIoU={best_miou:.2f}%  Output: {out_dir}")

if __name__=="__main__": main()
