#!/usr/bin/env python3
"""Train Plan9: F0'+L / F0 + DSM edge/slope prompt.

--variant A: F0'+L base (FrozenSAM3DFMLoRA + 4xSEFusion)
--variant B: F0 base (MFNetSAM3 + 1xSEFusion)

Both add edge/slope channels to DSM input via 1x1 conv.
"""

from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
PHASE1 = f"{BASE}/Personal-Project/RS-SAM3-p6/phase1_mm_adapter"
F0P_DIR = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0p_frozen_baseline"
F0_DIR = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/f0_mfnet_sam3"
SHARED = f"{BASE}/Personal-Project/RS-SAM3-p6/phase4_mfnet_ablation/shared"
sys.path.extend([SE, PHASE1, F0P_DIR, F0_DIR, SHARED])

from dataset_adapter import (IGNORE_INDEX, NUM_CLASSES, VAIHINGEN_TRAIN, VAIHINGEN_VAL,
                              POTSDAM_TRAIN, POTSDAM_VAL, _rgb_to_class)
from structure_loss import structure_loss

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]


def dataset_paths(dataset):
    if dataset == "vaihingen":
        tiles_train, tiles_val = VAIHINGEN_TRAIN, VAIHINGEN_VAL
        img_dir = "/root/autodl-tmp/dataset/Vaihingen/top"
        gt_dir = "/root/autodl-tmp/dataset/Vaihingen/gts_for_participants"
        img_suffix, gt_suffix = ".tif", ".tif"
        dsm_dir = "/root/autodl-tmp/dataset/Vaihingen/dsm"
        dsm_paths = {t: f'{dsm_dir}/dsm_09cm_matching_area{t.replace("top_mosaic_09cm_area", "")}.tif'
                     for t in tiles_train + tiles_val}
    else:
        tiles_train, tiles_val = POTSDAM_TRAIN, POTSDAM_VAL
        img_dir = "/root/autodl-tmp/dataset/Potsdam/2_Ortho_RGB"
        gt_dir = "/root/autodl-tmp/dataset/Potsdam/5_Labels_for_participants"
        img_suffix, gt_suffix = "_RGB.tif", "_label.tif"
        dsm_dir = "/root/autodl-tmp/dataset/Potsdam/1_DSM"
        dsm_paths = {t: f'{dsm_dir}/dsm_potsdam_{t.replace("top_potsdam_", "")}.tif'
                     for t in tiles_train + tiles_val}
    return tiles_train, tiles_val, img_dir, gt_dir, img_suffix, gt_suffix, dsm_paths


def load_sam3():
    prev = os.getcwd(); os.chdir(SE)
    from sam3 import build_sam3_image_model
    m = build_sam3_image_model(bpe_path=f"{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                                checkpoint_path=f"{SE}/weights/sam3/sam3.pt", device="cuda")
    os.chdir(prev)
    return m.cuda()


def make_edge_slope(dsm):
    """Compute edge (Laplacian) + slope (Sobel magnitude) from DSM. Input B×C×H×W or B×H×W."""
    if dsm.dim() == 4:
        dsm_sq = dsm[:, 0]  # B×H×W, take first channel
    else:
        dsm_sq = dsm
    B, H, W = dsm_sq.shape

    # Edge: Laplacian via conv2d
    lap_k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=dsm.device).view(1, 1, 3, 3)
    edge = torch.abs(F.conv2d(dsm_sq.view(B, 1, H, W), lap_k, padding=1)).view(B, H, W)

    # Slope: sqrt(dx^2 + dy^2), same size as input
    dx = dsm_sq[:, :, 1:] - dsm_sq[:, :, :-1]   # B×H×(W-1)
    dy = dsm_sq[:, 1:, :] - dsm_sq[:, :-1, :]   # B×(H-1)×W
    # Pad back to H×W
    dx = F.pad(dx, (1, 0))    # pad left
    dy = F.pad(dy, (0, 0, 1, 0))  # pad top
    slope = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

    return torch.stack([edge, slope], dim=1)  # B×2×H×W


class Plan9ModelA(nn.Module):
    """F0'+L + 3ch DSM (DSM + edge + slope) — no compression, 3 independent channels."""

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.resolution = base_model.resolution
        self.backbone = base_model.backbone
        self.pyramid_x = base_model.pyramid_x
        self.pyramid_y = base_model.pyramid_y
        self.decoder = base_model.decoder
        self.fusion4 = base_model.fusion4
        self.fusion3 = base_model.fusion3
        self.fusion2 = base_model.fusion2
        self.fusion1 = base_model.fusion1

    def forward(self, images, dsm):
        images = (images - 0.5) / 0.5
        if images.shape[-2:] != (self.resolution, self.resolution):
            images = F.interpolate(images, (self.resolution, self.resolution), mode="bilinear", align_corners=False)
        if dsm.dim() == 3:
            dsm_1ch_raw = dsm.unsqueeze(1)  # B×H×W → B×1×H×W
        elif dsm.dim() == 4 and dsm.shape[1] > 1:
            dsm_1ch_raw = dsm[:, :1]  # take first channel as raw DSM
        else:
            dsm_1ch_raw = dsm  # already B×1×H×W
        if dsm_1ch_raw.shape[-2:] != (self.resolution, self.resolution):
            dsm_1ch_raw = F.interpolate(dsm_1ch_raw, (self.resolution, self.resolution), mode="bilinear", align_corners=False)

        # Build enriched 3ch DSM: raw DSM + edge + slope
        es = make_edge_slope(dsm_1ch_raw)
        dsm_3ch = torch.cat([dsm_1ch_raw, es], dim=1)  # B×3×H×W

        # RGB forward
        backbone_out_rgb = self.backbone.forward_image(images)
        deepx = backbone_out_rgb["backbone_fpn"][-1]

        # DSM forward with 3ch enriched input (skip repeat(1,3)!)
        backbone_out_dsm = self.backbone.forward_image(dsm_3ch)
        deepy = backbone_out_dsm["backbone_fpn"][-1]

        fx = self.pyramid_x(deepx); fy = self.pyramid_y(deepy)
        f4 = self.fusion4(fx[0], fy[0]); f3 = self.fusion3(fx[1], fy[1])
        f2 = self.fusion2(fx[2], fy[2]); f1 = self.fusion1(fx[3], fy[3])
        return self.decoder([f4, f3, f2, f1])


class Plan9ModelB(nn.Module):
    """F0 + 3ch DSM — same approach on shared encoder baseline."""

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.resolution = base_model.resolution
        self.backbone = base_model.backbone
        self.pyramid = base_model.pyramid
        self.fusion = base_model.fusion if hasattr(base_model, 'fusion') else None
        self.decoder = base_model.decoder

    def forward(self, images, dsm):
        images = (images - 0.5) / 0.5
        if images.shape[-2:] != (self.resolution, self.resolution):
            images = F.interpolate(images, (self.resolution, self.resolution), mode="bilinear", align_corners=False)
        if dsm.dim() == 3:
            dsm = dsm.unsqueeze(1)
        if dsm.shape[-2:] != (self.resolution, self.resolution):
            dsm = F.interpolate(dsm, (self.resolution, self.resolution), mode="bilinear", align_corners=False)

        es = make_edge_slope(dsm)
        dsm_3ch = torch.cat([dsm[:, :1], es], dim=1)  # B×3×H×W

        backbone_out_rgb = self.backbone.forward_image(images)
        deepx = backbone_out_rgb["backbone_fpn"][-1]
        backbone_out_dsm = self.backbone.forward_image(dsm_3ch)
        deepy = backbone_out_dsm["backbone_fpn"][-1]

        fused = self.fusion(deepx, deepy)
        feats = self.pyramid(fused)
        return self.decoder(feats)


class OnlineCropDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, gt_dir, tiles, img_suf, gt_suf, dsm_paths,
                 crop_size=256, epoch_steps=1000, batch_size=2):
        self.crop_size = crop_size; self.epoch_steps = epoch_steps
        self.batch_size = batch_size; self.tiles = tiles
        self.total = epoch_steps * batch_size
        self.cache = []
        for tile in tiles:
            ip = f"{img_dir}/{tile}{img_suf}"; gp = f"{gt_dir}/{tile}{gt_suf}"; dp = dsm_paths[tile]
            img = np.array(Image.open(ip).convert("RGB"))
            gt = _rgb_to_class(np.array(Image.open(gp).convert("RGB")))
            dsm = np.array(Image.open(dp)).astype(np.float32)
            dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
            self.cache.append((img, gt, dsm))
        print(f"  Cached {len(tiles)} tiles, online crops/epoch={self.total}")

    def __len__(self): return self.total
    def __getitem__(self, idx):
        img, gt, dsm = self.cache[np.random.randint(0, len(self.cache))]
        h, w = img.shape[:2]; cs = self.crop_size
        if h > cs and w > cs:
            y = np.random.randint(0, h - cs); x = np.random.randint(0, w - cs)
            patch, label, dsm_p = img[y:y + cs, x:x + cs], gt[y:y + cs, x:x + cs], dsm[y:y + cs, x:x + cs]
        else:
            patch, label, dsm_p = img, gt, dsm
        return (torch.from_numpy(patch.copy()).permute(2, 0, 1).float() / 255.0,
                torch.from_numpy(dsm_p.copy()).float(),
                torch.from_numpy(label.copy()).long())


def make_val_ds(img_dir, gt_dir, tiles, img_suf, gt_suf, dsm_paths, stride=128):
    samples = []
    for tile in tiles:
        ip = f"{img_dir}/{tile}{img_suf}"; gp = f"{gt_dir}/{tile}{gt_suf}"; dp = dsm_paths[tile]
        if not os.path.exists(ip): continue
        img = np.array(Image.open(ip).convert("RGB"))
        gt = _rgb_to_class(np.array(Image.open(gp).convert("RGB")))
        dsm = np.array(Image.open(dp)).astype(np.float32) if os.path.exists(dp) else np.zeros(img.shape[:2])
        dsm = (dsm - dsm.min()) / max(dsm.max() - dsm.min(), 1e-8)
        h, w = img.shape[:2]
        for y in range(0, h - 128, stride):
            for x in range(0, w - 128, stride):
                y2, x2 = min(y + 256, h), min(x + 256, w); ph, pw = y2 - y, x2 - x
                if ph < 128 or pw < 128: continue
                p, d, l = img[y:y2, x:x2], dsm[y:y2, x:x2], gt[y:y2, x:x2]
                if ph < 256 or pw < 256:
                    p = np.pad(p, ((0, 256 - ph), (0, 256 - pw), (0, 0)), mode="reflect")
                    l = np.pad(l, ((0, 256 - ph), (0, 256 - pw)), "constant", constant_values=IGNORE_INDEX)
                    d = np.pad(d, ((0, 256 - ph), (0, 256 - pw)), mode="reflect")
                if (l == IGNORE_INDEX).mean() <= 0.5: samples.append((p, d, l))
    print(f"  {len(samples)} val windows")
    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self, i):
            img, dsm, label = samples[i]
            return (torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0,
                    torch.from_numpy(dsm.copy()).float(),
                    torch.from_numpy(label.copy()).long())
    return DS()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    inter = torch.zeros(NUM_CLASSES, device=device); union = torch.zeros(NUM_CLASSES, device=device)
    correct, total = 0, 0
    for images, dsm, labels in loader:
        images, dsm, labels = images.to(device), dsm.to(device), labels.to(device)
        logits = model(images, dsm)
        logits = F.interpolate(logits, labels.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1); mask = labels != IGNORE_INDEX
        correct += (pred[mask] == labels[mask]).sum().item(); total += mask.sum().item()
        for c in range(NUM_CLASSES):
            pc, lc = pred == c, labels == c; inter[c] += (pc & lc).sum(); union[c] += (pc | lc).sum()
    pci = {CLASS_NAMES[c]: (inter[c] / union[c].clamp(min=1) * 100).item() for c in range(NUM_CLASSES)}
    return {"avg_oa": correct / max(total, 1) * 100, "avg_miou": float(np.mean(list(pci.values()))),
            "per_class_iou": pci}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["A", "B"])
    parser.add_argument("--dataset", default="vaihingen")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epoch-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--output", default="/root/autodl-tmp/runs")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output, f"plan9_p9{args.variant}_{args.dataset}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    print(f"Plan9-{args.variant}: LoRA + DSM edge/slope prompt")
    print(f"  Dataset: {args.dataset} | epochs={args.epochs} | batch={args.batch}")

    train_t, val_t, img_dir, gt_dir, img_suf, gt_suf, dsm_paths = dataset_paths(args.dataset)
    train_ds = OnlineCropDataset(img_dir, gt_dir, train_t, img_suf, gt_suf, dsm_paths,
                                 epoch_steps=args.epoch_steps, batch_size=args.batch)
    val_ds = make_val_ds(img_dir, gt_dir, val_t, img_suf, gt_suf, dsm_paths)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    print("\nBuilding model...")
    sam3 = load_sam3()

    if args.variant == "A":
        CKPT = "/root/autodl-tmp/runs/plan6_phase4_f0p_lora_vaihingen_20260520_093831/best_model.pt"
        from model_f0p_lora import FrozenSAM3DFMLoRA
        base = FrozenSAM3DFMLoRA(sam3, num_classes=NUM_CLASSES, dropout=0.1, resolution=args.resolution).cuda()
        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        base.load_state_dict(ckpt["model"], strict=True)
        model = Plan9ModelA(base)
    else:
        CKPT = "/root/autodl-tmp/runs/plan6_phase4_f0_vaihingen_20260518_213923/best_model.pt"
        from model_f0 import MFNetSAM3
        base = MFNetSAM3(sam3, num_classes=NUM_CLASSES, dropout=0.1, resolution=args.resolution).cuda()
        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        base.load_state_dict(ckpt["model"], strict=True)
        model = Plan9ModelB(base)

    print(f"  Init from: {CKPT}")
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_miou = 0.0
    history = {"loss": [], "metrics": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        model.train(); start = time.time(); running_loss = 0.0
        for step, (images, dsm, labels) in enumerate(train_loader):
            images, dsm, labels = images.to(device), dsm.to(device), labels.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images, dsm)
                logits = F.interpolate(logits, labels.shape[-2:], mode="bilinear", align_corners=False)
                loss = structure_loss(logits.float(), labels)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
            running_loss += loss.item()
            if step % 100 == 0:
                print(f"  E{epoch:03d}/{args.epochs} B{step:04d} loss={loss.item():.4f}")

        sch.step(); avg_loss = running_loss / max(len(train_loader), 1)
        metrics = validate(model, val_loader, device)
        print(f"  E{epoch:03d}: loss={avg_loss:.4f} OA={metrics['avg_oa']:.2f}% mIoU={metrics['avg_miou']:.2f}% "
              f"best={best_miou:.2f}% time={time.time() - start:.0f}s")
        if metrics["avg_miou"] > best_miou:
            best_miou = metrics["avg_miou"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_v": best_miou, "metrics": metrics,
                        "args": vars(args)}, os.path.join(out_dir, "best_model.pt"))

        history["loss"].append(avg_loss); history["metrics"].append(metrics)
        json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)

    print(f"\nDone. Best mIoU={best_miou:.2f}% | Run: {out_dir}")


if __name__ == "__main__":
    main()
