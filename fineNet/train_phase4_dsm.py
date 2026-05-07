#!/usr/bin/env python3
"""
Plan1 Phase 4: Dual-Stream RGB + DSM Fine-tuning

SAM3 RGB features + lightweight CNN DSM encoder → UNet decoder.
"""

import os, sys, json, time, gc, random, argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

SE = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, SE)
_prev_cwd = os.getcwd(); os.chdir(SE)
from sam3 import build_sam3_image_model
os.chdir(_prev_cwd)

# === DSM Encoder ===
class DSMEncoder(nn.Module):
    def __init__(self, out_ch=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1,32,7,stride=2,padding=3),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
            nn.Conv2d(64,out_ch,3,stride=2,padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),
        )
    def forward(self, dsm):
        if dsm.dim()==3: dsm=dsm.unsqueeze(1)
        x=self.enc(dsm)  # (B,64,H/8,W/8)
        scales=[]
        for s in [288,144,72]:
            scales.append(F.interpolate(x,(s,s),mode='bilinear',align_corners=False))
        return scales  # match SAM3 FPN scales

# === Dual Decoder ===
class ConvBlock(nn.Module):
    def __init__(self,i,o):
        super().__init__()
        self.c=nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(inplace=True),
                             nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(inplace=True))
    def forward(self,x): return self.c(x)

class DualDecoder(nn.Module):
    def __init__(self,nc=5,dc=64):
        super().__init__()
        ic=256+dc
        self.u2=nn.ConvTranspose2d(ic,256,2,stride=2); self.c2=ConvBlock(256+ic,256)
        self.u1=nn.ConvTranspose2d(256,256,2,stride=2); self.c1=ConvBlock(256+ic,256)
        self.h=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(256,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(128,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
            nn.Conv2d(64,nc,1))
    def forward(self,rf,df):
        f=[torch.cat([r,d],dim=1) for r,d in zip(rf,df)]  # [320ch at each scale]
        x=self.u2(f[2])
        if x.shape[-2:]!=f[1].shape[-2:]: x=F.interpolate(x,f[1].shape[-2:],mode='bilinear',align_corners=False)
        x=self.c2(torch.cat([x,f[1]],dim=1))
        x=self.u1(x)
        if x.shape[-2:]!=f[0].shape[-2:]: x=F.interpolate(x,f[0].shape[-2:],mode='bilinear',align_corners=False)
        x=self.c1(torch.cat([x,f[0]],dim=1))
        return self.h(x)

# === SAM3 Extractor ===
class SAM3Extractor:
    def __init__(self, device='cuda'):
        _cwd=os.getcwd(); os.chdir(SE)
        self.model=build_sam3_image_model(bpe_path=f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz',
                                          checkpoint_path=f'{SE}/weights/sam3/sam3.pt',device='cuda')
        os.chdir(_cwd)
        for p in self.model.parameters(): p.requires_grad=False
    @torch.no_grad()
    def extract(self,images):
        xn=(images-0.5)/0.5; _,_,h,w=xn.shape
        if h!=1008 or w!=1008: xn=F.interpolate(xn,(1008,1008),mode='bilinear',align_corners=False)
        out=self.model.backbone.forward_image(xn)
        fpn=out['backbone_fpn']
        return [fpn[k].detach() for k in sorted(fpn.keys())] if isinstance(fpn,dict) else [f.detach() for f in fpn[:3]]
    def cleanup(self): del self.model; gc.collect(); torch.cuda.empty_cache()

# === Dataset with DSM ===
class DSMTrainDataset(torch.utils.data.Dataset):
    LABEL_MAP={1:0,2:1,3:2,4:3,5:4,6:255}
    def __init__(self,img_d,gt_d,dsm_d,tiles,img_s,gt_s,dsm_pat,crop=512,train=True):
        self.samples=[]
        for t in tiles:
            ip=os.path.join(img_d,f'{t}{img_s}'); gp=os.path.join(gt_d,f'{t}{gt_s}')
            if not os.path.exists(ip) or not os.path.exists(gp): continue
            dp=None
            for v in [t,t.replace('top_mosaic_09cm_',''),t.replace('top_potsdam_','potsdam_')]:
                c=os.path.join(dsm_d,dsm_pat.format(tile=v)); 
                if os.path.exists(c): dp=c; break
            self.samples.append((ip,gp,dp))
        self.crop,self.train,self.n=crop,train,80 if train else 10
        print(f"  {len(self.samples)} tiles ({sum(1 for _,_,d in self.samples if d)} with DSM)")
    def __len__(self): return len(self.samples)*self.n
    def __getitem__(self,idx):
        ip,gp,dp=self.samples[idx%len(self.samples)]
        img=np.array(Image.open(ip).convert('RGB'))
        label=np.array(Image.open(gp))
        rm=np.full_like(label,255,dtype=np.int64)
        for k,v in self.LABEL_MAP.items(): rm[label==k]=v
        dsm=np.array(Image.open(dp)).astype(np.float32) if dp else np.zeros(img.shape[:2],dtype=np.float32)
        from scipy.ndimage import grey_opening
        try:
            ndsm=dsm-grey_opening(dsm,size=101); dsm_n=np.clip(ndsm/10.,0,1)
        except: dsm_n=dsm/10.
        if self.train:
            h,w=img.shape[:2]
            if h>self.crop and w>self.crop:
                y=random.randint(0,h-self.crop); x=random.randint(0,w-self.crop)
                img=img[y:y+self.crop,x:x+self.crop]; rm=rm[y:y+self.crop,x:x+self.crop]; dsm_n=dsm_n[y:y+self.crop,x:x+self.crop]
            if random.random()<0.5: img=np.fliplr(img).copy(); rm=np.fliplr(rm).copy(); dsm_n=np.fliplr(dsm_n).copy()
            if random.random()<0.5: img=np.flipud(img).copy(); rm=np.flipud(rm).copy(); dsm_n=np.flipud(dsm_n).copy()
            k=random.randint(0,3)
            if k>0: img=np.rot90(img,k).copy(); rm=np.rot90(rm,k).copy(); dsm_n=np.rot90(dsm_n,k).copy()
        return torch.from_numpy(img).permute(2,0,1).float()/255.,torch.from_numpy(dsm_n).float(),torch.from_numpy(rm).long()

# === Validation ===
@torch.no_grad()
def validate(decoder,dsm_enc,extractor,loader,device):
    decoder.eval(); dsm_enc.eval()
    inter=torch.zeros(5,device=device); union=torch.zeros(5,device=device); corr=0; tot=0
    for img,dsm,lab in loader:
        img,dsm,lab=img.to(device),dsm.to(device),lab.to(device)
        rf=extractor.extract(img); df=dsm_enc(dsm)
        logits=decoder(rf,df)
        logits=F.interpolate(logits,lab.shape[-2:],mode='bilinear',align_corners=False)
        pred=logits.argmax(1); mask=(lab!=255)
        corr+=(pred[mask]==lab[mask]).sum().item(); tot+=mask.sum().item()
        for c in range(5): pc=(pred==c); lc=(lab==c); inter[c]+=(pc&lc).sum(); union[c]+=(pc|lc).sum()
    aAcc=corr/max(tot,1)*100
    pc=[(inter[c]/max(union[c],1)*100).item() for c in range(5)]
    return aAcc,np.mean(pc),pc

# === Main ===
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch',type=int,default=2)
    p.add_argument('--crop',type=int,default=512); p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--output',default='/root/Mynet/autodl-tmp/runs')
    args=p.parse_args()
    device=torch.device('cuda')
    ts=datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir=os.path.join(args.output,f'phase4_dsm_{ts}')
    os.makedirs(out_dir,exist_ok=True)
    # Models
    extractor=SAM3Extractor(device)
    dsm_enc=DSMEncoder(64).to(device); decoder=DualDecoder(5,64).to(device)
    print(f"DSM: {sum(p.numel() for p in dsm_enc.parameters()):,} | Dec: {sum(p.numel() for p in decoder.parameters()):,}")
    # Data
    V='/root/autodl-tmp/dataset/Vaihingen'; P='/root/autodl-tmp/dataset/Potsdam'
    v_tr=[f'top_mosaic_09cm_area{i}' for i in [1,3,5,7,11,13,15,17,21,23,26,28]]
    v_vl=[f'top_mosaic_09cm_area{i}' for i in [30,32,34,37]]
    p_tr=[f'top_potsdam_{t}_{n}' for t in ['2','3','4','5'] for n in ['10','11','12']]+[f'top_potsdam_{t}_{n}' for t in ['6','7'] for n in ['10','11']]
    p_vl=[f'top_potsdam_{t}_{n}' for t in ['6','7'] for n in ['7','8','9','12']]
    tr_ds=torch.utils.data.ConcatDataset([
        DSMTrainDataset(V+'/top',V+'/gts_index',V+'/dsm',v_tr,'.tif','.png','dsm_09cm_matching_{tile}.tif',args.crop,True),
        DSMTrainDataset(P+'/2_Ortho_RGB',P+'/labels_index',P+'/1_DSM',p_tr,'_RGB.tif','.png','dsm_{tile}.tif',args.crop,True),
    ])
    v_ldr=torch.utils.data.DataLoader(DSMTrainDataset(V+'/top',V+'/gts_index',V+'/dsm',v_vl,'.tif','.png','dsm_09cm_matching_{tile}.tif',args.crop,False),1,num_workers=0)
    p_ldr=torch.utils.data.DataLoader(DSMTrainDataset(P+'/2_Ortho_RGB',P+'/labels_index',P+'/1_DSM',p_vl,'_RGB.tif','.png','dsm_{tile}.tif',args.crop,False),1,num_workers=0)
    tr_ldr=torch.utils.data.DataLoader(tr_ds,args.batch,shuffle=True,num_workers=0,drop_last=True)
    # Opt
    all_p=list(dsm_enc.parameters())+list(decoder.parameters())
    opt=torch.optim.AdamW(all_p,lr=args.lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    scaler=torch.amp.GradScaler('cuda'); crit=nn.CrossEntropyLoss(ignore_index=255)
    best_v,best_p=0.,0.; hist={'loss':[],'v_miou':[],'p_miou':[]}
    print(f"\nTraining {args.epochs} epochs...")
    for epoch in range(1,args.epochs+1):
        decoder.train(); dsm_enc.train(); el,nb=0.,0; t0=time.time()
        for bi,(img,dsm,lab) in enumerate(tr_ldr):
            img,dsm,lab=img.to(device),dsm.to(device),lab.to(device)
            rf=extractor.extract(img); df=dsm_enc(dsm)
            logits=decoder(rf,df); logits=F.interpolate(logits,lab.shape[-2:],mode='bilinear',align_corners=False)
            loss=crit(logits,lab)
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(all_p,1.0)
            scaler.step(opt); scaler.update()
            el+=loss.item(); nb+=1
            if bi%20==0: print(f"  E{epoch}/{args.epochs} B{bi}/{len(tr_ldr)} loss={loss.item():.4f}",end='\r',flush=True)
        sch.step(); avg=el/max(nb,1)
        va,vm,vpc=validate(decoder,dsm_enc,extractor,v_ldr,device)
        pa,pm,ppc=validate(decoder,dsm_enc,extractor,p_ldr,device)
        hist['loss'].append(avg); hist['v_miou'].append(vm); hist['p_miou'].append(pm)
        print(f"  E{epoch:3d}: loss={avg:.4f} V={vm:.1f}% P={pm:.1f}%  ({time.time()-t0:.0f}s)",flush=True)
        if vm>best_v: best_v=vm; torch.save({'epoch':epoch,'dsm_enc':dsm_enc.state_dict(),'decoder':decoder.state_dict(),'v_miou':vm,'p_miou':pm,'v_per_class':vpc,'p_per_class':ppc},os.path.join(out_dir,'best_model.pt'))
        json.dump(hist,open(os.path.join(out_dir,'history.json'),'w'))
    extractor.cleanup()
    print(f"\nDone! Best V={best_v:.1f}% P={best_p:.1f}%\nOutput: {out_dir}",flush=True)

if __name__=='__main__': main()
