#!/usr/bin/env python3
"""B1 ablation: A3 + softmax gate (vs sigmoid). Hot-start from A3 checkpoint."""
import json, os, sys, time, numpy as np, torch, torch.nn.functional as F
from datetime import datetime
BASE = "/root/Mynet"; SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
sys.path.insert(0, SE); sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p6/phase3_ablation")
sys.path.insert(0, f"{BASE}/Personal-Project/RS-SAM3-p6/phase3_ablation/chain1_adapter")

# Import model with softmax gates
import mm_adapter_vit_softmax as mma_soft
from simple_decoder import SimpleDecoder

class Pyramid4Scale(torch.nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.up2 = torch.nn.Sequential(torch.nn.ConvTranspose2d(ch,ch,2,2),torch.nn.BatchNorm2d(ch),torch.nn.ReLU(inplace=False))
        self.up4 = torch.nn.Sequential(torch.nn.ConvTranspose2d(ch,ch,2,2),torch.nn.BatchNorm2d(ch),torch.nn.ReLU(inplace=False),
                                       torch.nn.ConvTranspose2d(ch,ch,2,2),torch.nn.BatchNorm2d(ch),torch.nn.ReLU(inplace=False))
        self.id3 = torch.nn.Identity(); self.down = torch.nn.MaxPool2d(2,2)
    def forward(self, feat): return [self.up4(feat),self.up2(feat),self.id3(feat),self.down(feat)]

class B1SoftmaxGate(torch.nn.Module):
    def __init__(self, sam3_model, num_classes=5, bottleneck=32, dsm_dim=128):
        super().__init__()
        self.backbone = sam3_model.backbone; self.resolution = 1008
        self.mm_state = mma_soft.inject_mm_adapters(self.backbone.vision_backbone, bottleneck=bottleneck,
                                                     dsm_attn_mode="full", checkpoint_attn=True)
        self.dsm_encoder = mma_soft.DSMTokenEncoder(token_dim=1024, dsm_dim=dsm_dim)
        self.pyramid = Pyramid4Scale(256)
        self.decoder = SimpleDecoder(in_channels=256, decode_channels=64, num_classes=num_classes)
        if hasattr(self.backbone, "language_backbone"):
            for p in self.backbone.language_backbone.parameters(): p.requires_grad = False
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  B1 softmax: {t:,} trainable / {total:,} total")

    def forward(self, images, dsm):
        images = (images - 0.5) / 0.5
        if images.shape[-2:] != (self.resolution, self.resolution):
            images = F.interpolate(images, (self.resolution,self.resolution), mode="bilinear", align_corners=False)
        if dsm.dim() == 3: dsm = dsm.unsqueeze(1)
        if dsm.shape[-2:] != (self.resolution, self.resolution):
            dsm = F.interpolate(dsm, (self.resolution,self.resolution), mode="bilinear", align_corners=False)
        patch_hw = (self.resolution // 16, self.resolution // 16)
        self.mm_state["dsm_tokens"] = self.dsm_encoder(dsm, patch_hw)
        try: out = self.backbone.forward_image(images)
        finally: self.mm_state.pop("dsm_tokens", None)
        feat = out["backbone_fpn"][-1].clone()
        scales = self.pyramid(feat)
        logits = self.decoder(scales[-1])
        return F.interpolate(logits, (256,256), mode="bilinear", align_corners=False)


from dataset_adapter import VAIHINGEN_TRAIN, VAIHINGEN_VAL, _rgb_to_class, IGNORE_INDEX
from dataset_online import OnlineCropDataset
from structure_loss import structure_loss
from PIL import Image
NUM_CLASSES=5; CLASS_NAMES=["road","building","grass","tree","car"]

def load_sam3():
    prev=os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    m=build_sam3_image_model(bpe_path=f"{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                              checkpoint_path=f"{SE}/weights/sam3/sam3.pt",device="cuda")
    os.chdir(prev); return m.cuda()

def make_val_ds():
    tiles=VAIHINGEN_VAL; samples=[]
    dsm_dir="/root/autodl-tmp/dataset/Vaihingen/dsm"
    for tile in tiles:
        ip=f"/root/autodl-tmp/dataset/Vaihingen/top/{tile}.tif"
        gp=f"/root/autodl-tmp/dataset/Vaihingen/gts_for_participants/{tile}.tif"
        dp=f'{dsm_dir}/dsm_09cm_matching_area{tile.replace("top_mosaic_09cm_area","")}.tif'
        if not os.path.exists(ip): continue
        img=np.array(Image.open(ip).convert("RGB"))
        gt=_rgb_to_class(np.array(Image.open(gp).convert("RGB")))
        dsm=np.array(Image.open(dp) if os.path.exists(dp) else np.zeros(img.shape[:2])).astype(np.float32)
        dsm=(dsm-dsm.min())/max(dsm.max()-dsm.min(),1e-8)
        h,w=img.shape[:2]
        for y in range(0,h-128,128):
            for x in range(0,w-128,128):
                y2,x2=min(y+256,h),min(x+256,w); ph,pw=y2-y,x2-x
                if ph<128 or pw<128: continue
                p,d,l=img[y:y2,x:x2],dsm[y:y2,x:x2],gt[y:y2,x:x2]
                if ph<256 or pw<256:
                    p=np.pad(p,((0,256-ph),(0,256-pw),(0,0)),mode="reflect")
                    l=np.pad(l,((0,256-ph),(0,256-pw)),"constant",constant_values=IGNORE_INDEX)
                    d=np.pad(d,((0,256-ph),(0,256-pw)),mode="reflect")
                if (l==IGNORE_INDEX).mean()<=0.5: samples.append((p,d,l))
    print(f"  {len(samples)} val windows")
    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self,i):
            img,dsm,label=samples[i]
            return torch.from_numpy(img.copy()).permute(2,0,1).float()/255.0,torch.from_numpy(dsm.copy()).float(),torch.from_numpy(label.copy()).long()
    return DS()

@torch.no_grad()
def validate(model,loader,device):
    model.eval(); inter=torch.zeros(NUM_CLASSES,device=device); union=torch.zeros(NUM_CLASSES,device=device)
    correct=0; total=0
    for images,dsm,labels in loader:
        images,dsm,labels=images.to(device),dsm.to(device),labels.to(device)
        logits=model(images,dsm)
        logits=F.interpolate(logits,labels.shape[-2:],mode="bilinear",align_corners=False)
        pred=logits.argmax(1); mask=labels!=IGNORE_INDEX
        correct+=(pred[mask]==labels[mask]).sum().item(); total+=mask.sum().item()
        for c in range(NUM_CLASSES):
            pc,lc=pred==c,labels==c; inter[c]+=(pc&lc).sum(); union[c]+=(pc|lc).sum()
    pci={CLASS_NAMES[c]:(inter[c]/union[c].clamp(min=1)*100).item() for c in range(NUM_CLASSES)}
    return {"avg_oa":correct/max(total,1)*100,"avg_miou":float(np.mean(list(pci.values()))),"per_class_iou":pci}

def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--batch",type=int,default=2); p.add_argument("--lr",type=float,default=5e-5)
    p.add_argument("--output",default="/root/autodl-tmp/runs"); args=p.parse_args()
    device="cuda"; ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir=os.path.join(args.output,f"plan6_phase3_b1_{ts}"); os.makedirs(out_dir,exist_ok=True)
    print(f"B1: softmax gate, {args.epochs} epochs")

    tiles=VAIHINGEN_TRAIN
    dsm_dir="/root/autodl-tmp/dataset/Vaihingen/dsm"
    dsm_paths={t:f'{dsm_dir}/dsm_09cm_matching_area{t.replace("top_mosaic_09cm_area","")}.tif' for t in tiles}
    train_ds=OnlineCropDataset("/root/autodl-tmp/dataset/Vaihingen/top","/root/autodl-tmp/dataset/Vaihingen/gts_for_participants",
                               tiles,".tif",".tif",dsm_paths,is_train=True,crop_size=256,epoch_steps=1000,batch_size=args.batch)
    val_ds=make_val_ds()
    train_loader=torch.utils.data.DataLoader(train_ds,batch_size=args.batch,shuffle=False,num_workers=0,drop_last=False,pin_memory=True)
    val_loader=torch.utils.data.DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=0)

    print("\nBuilding B1 with A3 init...")
    sam3=load_sam3()
    model=B1SoftmaxGate(sam3,num_classes=5).cuda()
    ckpt=torch.load("/root/autodl-tmp/runs/plan6_phase3_a3_20260513_204507/best_model.pt",map_location="cpu",weights_only=False)
    # sigmoid gate params won't match softmax; load with strict=False
    missing,unexpected=model.load_state_dict(ckpt["model"],strict=False)
    print(f"  Loaded from A3: {len(missing)} missing, {len(unexpected)} unexpected")
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    scaler=torch.amp.GradScaler("cuda")
    best_miou=0.0; history={"loss":[],"metrics":[]}

    for epoch in range(1,args.epochs+1):
        model.train(); start=time.time(); run_loss=0.0
        for step,(images,dsm,labels) in enumerate(train_loader):
            images,dsm,labels=images.to(device),dsm.to(device),labels.to(device)
            with torch.amp.autocast("cuda",dtype=torch.bfloat16):
                logits=model(images,dsm)
                logits=F.interpolate(logits,labels.shape[-2:],mode="bilinear",align_corners=False)
                loss=structure_loss(logits.float(),labels)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
            run_loss+=loss.item()
        sch.step(); avg_loss=run_loss/max(len(train_loader),1)
        metrics=validate(model,val_loader,device)
        print(f"  B1 E{epoch:02d}: loss={avg_loss:.4f} OA={metrics['avg_oa']:.2f}% mIoU={metrics['avg_miou']:.2f}% best={best_miou:.2f}%")
        if metrics["avg_miou"]>best_miou:
            best_miou=metrics["avg_miou"]
            torch.save({"epoch":epoch,"model":model.state_dict(),"best_v":best_miou,"metrics":metrics},os.path.join(out_dir,"best_model.pt"))
        history["loss"].append(avg_loss); history["metrics"].append(metrics)
        json.dump(history,open(os.path.join(out_dir,"history.json"),"w"),indent=2)
    print(f"B1 done. best={best_miou:.2f}%")

if __name__=="__main__": main()
