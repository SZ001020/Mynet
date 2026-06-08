#!/usr/bin/env python3
"""P13-G: strict vegetation-only multi-scale texture refinement.

Five intended variants are launched by run_g_experiments.sh:
G1 oracle RGB texture, G2 pred RGB texture, G3 oracle RGB+nDSM roughness,
G4 pred RGB+nDSM roughness, G5 pred RGB+nDSM roughness with DiceCE.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = "/root/Mynet"
SE = f"{BASE}/Reference-Project/SegEarth-OV-3-main"
P7_DIR = f"{BASE}/Personal-Project/RS-SAM3-p7/phase_a_dsm_prompt"
P11B_DIR = f"{BASE}/Personal-Project/RS-SAM3-p11/phase_b_ndsm"
P13G_DIR = f"{BASE}/Personal-Project/RS-SAM3-p13/phase_g_strict_veg_texture"
sys.path.insert(0, SE)
sys.path.insert(0, P7_DIR)
sys.path.insert(0, P11B_DIR)
sys.path.insert(0, P13G_DIR)

from dataset_adapter import (  # noqa: E402
    IGNORE_INDEX,
    NUM_CLASSES,
    POTSDAM_TRAIN,
    POTSDAM_VAL,
    VAIHINGEN_TRAIN,
    VAIHINGEN_VAL,
    _rgb_to_class,
)
from dataset_online import OnlineCropDataset, compute_ndsm  # noqa: E402

_p7_spec = importlib.util.spec_from_file_location("p7_model", f"{P7_DIR}/model.py")
_p7_mod = importlib.util.module_from_spec(_p7_spec)
assert _p7_spec.loader is not None
_p7_spec.loader.exec_module(_p7_mod)
Plan7PromptMFNet = _p7_mod.Plan7PromptMFNet

_spec = importlib.util.spec_from_file_location("p13g_model", f"{P13G_DIR}/model.py")
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
Plan13GStrictVegTexture = _mod.Plan13GStrictVegTexture

CLASS_NAMES = ["road", "building", "grass", "tree", "car"]


class Window256DatasetDSM(torch.utils.data.Dataset):
    def __init__(self, img_dir, gt_dir, tiles, img_suffix, gt_suffix, dsm_paths, stride=128):
        self.samples = []
        for tile in tiles:
            ip, gp, dp = f"{img_dir}/{tile}{img_suffix}", f"{gt_dir}/{tile}{gt_suffix}", dsm_paths.get(tile)
            if not os.path.exists(ip) or not os.path.exists(gp):
                continue
            img = np.array(Image.open(ip).convert("RGB"))
            gt = _rgb_to_class(np.array(Image.open(gp).convert("RGB")))
            if dp and os.path.exists(dp):
                dsm = compute_ndsm(np.array(Image.open(dp)).astype(np.float32))
            else:
                dsm = np.zeros(img.shape[:2], dtype=np.float32)
            h, w = img.shape[:2]
            for y in range(0, h - 128, stride):
                for x in range(0, w - 128, stride):
                    y2, x2 = min(y + 256, h), min(x + 256, w)
                    ph, pw = y2 - y, x2 - x
                    if ph < 128 or pw < 128:
                        continue
                    patch, label, dsm_patch = img[y:y2, x:x2], gt[y:y2, x:x2], dsm[y:y2, x:x2]
                    if ph < 256 or pw < 256:
                        patch = np.pad(patch, ((0, 256 - ph), (0, 256 - pw), (0, 0)), mode="reflect")
                        label = np.pad(label, ((0, 256 - ph), (0, 256 - pw)), mode="constant", constant_values=IGNORE_INDEX)
                        dsm_patch = np.pad(dsm_patch, ((0, 256 - ph), (0, 256 - pw)), mode="reflect")
                    if (label == IGNORE_INDEX).mean() <= 0.5:
                        self.samples.append((patch, dsm_patch, label))
        print(f"  {len(self.samples)} fixed RGB/DSM windows (val)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, dsm, label = self.samples[idx]
        return (
            torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0,
            torch.from_numpy(dsm.copy()).float(),
            torch.from_numpy(label.copy()).long(),
        )


def dataset_paths(dataset: str):
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


def load_sam3(device: str = "cuda"):
    prev = os.getcwd()
    os.chdir(SE)
    from sam3 import build_sam3_image_model

    model = build_sam3_image_model(
        bpe_path=f"{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        checkpoint_path=f"{SE}/weights/sam3/sam3.pt",
        device=device,
    )
    os.chdir(prev)
    return model.cuda()


def build_base_model(args):
    sam3 = load_sam3()
    base = Plan7PromptMFNet(
        sam3,
        adapter_bottleneck=args.adapter_bottleneck,
        num_classes=NUM_CLASSES,
        dropout=0.1,
        dsm_attn_mode=args.dsm_attn_mode,
        checkpoint_attn=False,
        resolution=args.resolution,
    ).cuda()
    ckpt = torch.load(args.init_from, map_location="cuda", weights_only=False)
    missing, unexpected = base.load_state_dict(ckpt["model"], strict=False)
    print(f"  Loaded P11-B base epoch={ckpt.get('epoch')} best={ckpt.get('best_v', 0):.2f}%")
    print(f"  Base missing={len(missing)} unexpected={len(unexpected)}")
    return base


def vegetation_loss(final_logits, labels, loss_type="ce"):
    veg_gt = (labels == 2) | (labels == 3)
    if veg_gt.sum() < 1:
        return None
    veg_logits = final_logits[:, 2:4]
    veg_logits = F.interpolate(veg_logits, labels.shape[-2:], mode="bilinear", align_corners=False)
    target = labels[veg_gt] - 2
    logits_2d = veg_logits.permute(0, 2, 3, 1)[veg_gt]
    ce = F.cross_entropy(logits_2d, target)
    if loss_type == "ce":
        return ce
    probs = torch.softmax(logits_2d.float(), dim=1)
    onehot = F.one_hot(target, num_classes=2).float()
    inter = (probs * onehot).sum(dim=0)
    denom = (probs + onehot).sum(dim=0).clamp_min(1e-6)
    dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
    return ce + dice


@torch.no_grad()
def validate(model, loader, device, mask_mode: str, max_batches: int | None = None):
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    invariant_errors = 0
    for batch_idx, (images, dsm, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, dsm, labels = images.to(device), dsm.to(device), labels.to(device)
        out = model(images, dsm, labels=labels, mask_mode=mask_mode)
        logits = F.interpolate(out["final_logits"], labels.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1)
        mask = labels != IGNORE_INDEX
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                cm[i, j] += ((labels == i) & (pred == j) & mask).sum()

        base = out["base_logits"]
        final = out["final_logits"]
        nonveg = (~out["veg_mask"]).unsqueeze(1).expand_as(final)
        invariant_errors += int((final[nonveg] != base[nonveg]).sum().item())

    cm_cpu = cm.cpu().numpy()
    ious, recalls = {}, {}
    for idx, name in enumerate(CLASS_NAMES):
        tp = cm_cpu[idx, idx]
        union = cm_cpu[idx, :].sum() + cm_cpu[:, idx].sum() - tp
        gt = cm_cpu[idx, :].sum()
        ious[name] = float(tp / max(union, 1) * 100)
        recalls[name] = float(tp / max(gt, 1) * 100)
    miou = float(np.mean(list(ious.values())))
    oa = float(np.trace(cm_cpu) / max(cm_cpu.sum(), 1) * 100)
    gt_err = int(cm_cpu[2, 3])
    tg_err = int(cm_cpu[3, 2])
    grass_gt = int(cm_cpu[2, :].sum())
    tree_gt = int(cm_cpu[3, :].sum())
    return {
        "mask_mode": mask_mode,
        "avg_miou": miou,
        "avg_oa": oa,
        "per_class_iou": ious,
        "per_class_recall": recalls,
        "confusion_matrix": cm_cpu.tolist(),
        "grass_to_tree": gt_err,
        "tree_to_grass": tg_err,
        "grass_to_tree_pct": gt_err / max(grass_gt, 1) * 100,
        "tree_to_grass_pct": tg_err / max(tree_gt, 1) * 100,
        "veg_sum": gt_err + tg_err,
        "invariant_errors": invariant_errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="g1_oracle_rgb")
    parser.add_argument("--dataset", default="vaihingen", choices=["vaihingen", "potsdam"])
    parser.add_argument("--init-from", default="/root/autodl-tmp/runs/plan11_b_ndsm_vaihingen_20260604_225146/best_model.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epoch-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--loss-type", default="ce", choices=["ce", "dicece"])
    parser.add_argument("--use-ndsm-roughness", action="store_true")
    parser.add_argument("--no-rgb-texture", action="store_true")
    parser.add_argument("--select-mask", default="oracle", choices=["oracle", "pred"])
    parser.add_argument("--adapter-bottleneck", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--dsm-attn-mode", default="full", choices=["adapter", "full"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--eval-epoch1", action="store_true",
                        help="Also run full validation after epoch 1; disabled by default for long formal runs")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="Limit validation batches for smoke tests; default uses all val windows")
    parser.add_argument("--output", default="/root/autodl-tmp/runs")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output, f"plan13_g_{args.experiment}_{args.dataset}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=2, default=str)
    print(f"P13-G strict vegetation texture: {args.experiment}")
    print(f"  Output: {out_dir}")

    train_t, val_t, img_dir, gt_dir, img_suf, gt_suf, dsm_paths = dataset_paths(args.dataset)
    train_ds = OnlineCropDataset(
        img_dir, gt_dir, train_t, img_suf, gt_suf, dsm_paths,
        is_train=True, crop_size=256, epoch_steps=args.epoch_steps, batch_size=args.batch,
    )
    val_ds = Window256DatasetDSM(img_dir, gt_dir, val_t, img_suf, gt_suf, dsm_paths)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda")
    base = build_base_model(args)
    model = Plan13GStrictVegTexture(
        base,
        resolution=args.resolution,
        use_rgb_texture=not args.no_rgb_texture,
        use_ndsm_roughness=args.use_ndsm_roughness,
        assert_invariant=True,
    ).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    history = {"loss": [], "metrics": [], "lr": []}
    best = -1.0
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.base_model.eval()
        start = time.time()
        running = 0.0
        train_steps = 0
        skipped_no_veg = 0
        for step, (images, dsm, labels) in enumerate(train_loader):
            images, dsm, labels = images.to(device), dsm.to(device), labels.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(images, dsm, labels=labels, mask_mode="oracle")
                loss = vegetation_loss(out["final_logits"], labels, args.loss_type)
            if loss is None:
                skipped_no_veg += 1
                if step % 100 == 0:
                    print(f"  E{epoch:03d}/{args.epochs} B{step:04d}/{len(train_loader)} skip=no_veg")
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            train_steps += 1
            if step % 100 == 0:
                print(f"  E{epoch:03d}/{args.epochs} B{step:04d}/{len(train_loader)} loss={loss.item():.4f}")

        scheduler.step()
        avg_loss = running / max(train_steps, 1)
        metrics = None
        should_validate = (epoch == args.epochs) or (args.val_every > 0 and epoch % args.val_every == 0)
        if args.eval_epoch1 and epoch == 1:
            should_validate = True
        if should_validate:
            oracle = validate(model, val_loader, device, "oracle", args.max_val_batches)
            pred = validate(model, val_loader, device, "pred", args.max_val_batches)
            metrics = {"oracle": oracle, "pred": pred}
            selected = metrics[args.select_mask]["avg_miou"]
            print(
                f"  E{epoch:03d}: loss={avg_loss:.4f} "
                f"steps={train_steps} skipped={skipped_no_veg} "
                f"oracle mIoU={oracle['avg_miou']:.2f} veg={oracle['veg_sum']} inv={oracle['invariant_errors']} | "
                f"pred mIoU={pred['avg_miou']:.2f} veg={pred['veg_sum']} inv={pred['invariant_errors']} "
                f"best={best:.2f} time={time.time() - start:.0f}s"
            )
            if selected > best:
                best = selected
                best_metrics = metrics
                torch.save(
                    {
                        "epoch": epoch,
                        "head": model.trainable_state_dict(),
                        "best_v": best,
                        "metrics": metrics,
                        "args": vars(args),
                    },
                    os.path.join(out_dir, "best_model.pt"),
                )
        history["loss"].append(avg_loss)
        history["metrics"].append(metrics)
        history["lr"].append(scheduler.get_last_lr())
        json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
        if best_metrics is not None:
            json.dump(best_metrics, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)

    print(f"\nDone. Best {args.select_mask} mIoU={best:.2f}%")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
