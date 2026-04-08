import csv
import itertools
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from skimage import io
from sklearn.metrics import confusion_matrix
from torch import amp
from torch.autograd import Variable
from tqdm.auto import tqdm

from UNetFormer_MMSAM import UNetFormer as MFNet
from modules.domain_discriminator import DomainAdversarialHead


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_from_color(arr_3d, palette):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for rgb, cls in palette.items():
        m = np.all(arr_3d == np.array(rgb).reshape(1, 1, 3), axis=2)
        arr_2d[m] = cls
    return arr_2d


def sliding_window(top, step=10, window_size=(20, 20)):
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    c = 0
    for _ in sliding_window(top, step=step, window_size=window_size):
        c += 1
    return c


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def cross_entropy_2d(input_, target, weight=None):
    output = input_.view(input_.size(0), input_.size(1), -1)
    output = torch.transpose(output, 1, 2).contiguous().view(-1, output.size(1))
    target = target.view(-1)
    return F.cross_entropy(output, target, weight=weight, reduction="mean")


ISPRS_PALETTE = {
    (255, 255, 255): 0,
    (0, 0, 255): 1,
    (0, 255, 255): 2,
    (0, 255, 0): 3,
    (255, 255, 0): 4,
    (255, 0, 0): 5,
    (0, 0, 0): 6,
}


def get_protocol(dataset: str, data_root: str):
    root = os.path.join(data_root, dataset)
    if dataset == "Vaihingen":
        train_ids = ["1", "3", "23", "26", "7", "11", "13", "28", "17", "32", "34", "37"]
        test_ids = ["5", "21", "15", "30"]
        data_t = os.path.join(root, "top", "top_mosaic_09cm_area{}.tif")
        dsm_t = os.path.join(root, "dsm", "dsm_09cm_matching_area{}.tif")
        label_t = os.path.join(root, "gts_for_participants", "top_mosaic_09cm_area{}.tif")
        eroded_t = os.path.join(root, "gts_eroded_for_participants", "top_mosaic_09cm_area{}_noBoundary.tif")
        stride = 32
    elif dataset == "Potsdam":
        train_ids = ["6_10", "7_10", "2_12", "3_11", "2_10", "7_8", "5_10", "3_12", "5_12", "7_11", "7_9", "6_9", "7_7", "4_12", "6_8", "6_12", "6_7", "4_11"]
        test_ids = ["4_10", "5_11", "2_11", "3_10", "6_11", "7_12"]
        data_t = os.path.join(root, "4_Ortho_RGBIR", "top_potsdam_{}_RGBIR.tif")
        dsm_t = os.path.join(root, "1_DSM_normalisation", "dsm_potsdam_{}_normalized_lastools.jpg")
        label_t = os.path.join(root, "5_Labels_for_participants", "top_potsdam_{}_label.tif")
        eroded_t = os.path.join(root, "5_Labels_for_participants_no_Boundary", "top_potsdam_{}_label_noBoundary.tif")
        stride = 128
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "data_t": data_t,
        "dsm_t": dsm_t,
        "label_t": label_t,
        "eroded_t": eroded_t,
        "stride": stride,
    }


class SourceDataset(torch.utils.data.Dataset):
    def __init__(self, proto: Dict, window_size: Tuple[int, int], epoch_steps: int, cache=True):
        self.ids = proto["train_ids"]
        self.data_t = proto["data_t"]
        self.dsm_t = proto["dsm_t"]
        self.label_t = proto["label_t"]
        self.window_size = window_size
        self.epoch_steps = epoch_steps
        self.cache = cache
        self.data_cache = {}
        self.dsm_cache = {}
        self.label_cache = {}

        for tile_id in self.ids:
            for p in [self.data_t.format(tile_id), self.dsm_t.format(tile_id), self.label_t.format(tile_id)]:
                if not os.path.isfile(p):
                    raise FileNotFoundError(p)

    def __len__(self):
        return self.epoch_steps

    def _augment(self, img, dsm, label):
        if random.random() < 0.5:
            img = img[:, ::-1, :]
            dsm = dsm[::-1, :]
            label = label[::-1, :]
        if random.random() < 0.5:
            img = img[:, :, ::-1]
            dsm = dsm[:, ::-1]
            label = label[:, ::-1]
        return np.copy(img), np.copy(dsm), np.copy(label)

    def __getitem__(self, idx):
        tile_id = random.choice(self.ids)
        if tile_id in self.data_cache:
            data = self.data_cache[tile_id]
            dsm = self.dsm_cache[tile_id]
            label = self.label_cache[tile_id]
        else:
            img = io.imread(self.data_t.format(tile_id))
            if img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]
            data = (img.transpose((2, 0, 1)) / 255.0).astype("float32")

            dsm_raw = np.asarray(io.imread(self.dsm_t.format(tile_id)), dtype="float32")
            dmin, dmax = np.min(dsm_raw), np.max(dsm_raw)
            dsm = ((dsm_raw - dmin) / (dmax - dmin + 1e-8)).astype("float32")

            label_rgb = np.asarray(io.imread(self.label_t.format(tile_id)))
            label = convert_from_color(label_rgb, ISPRS_PALETTE).astype("int64")

            if self.cache:
                self.data_cache[tile_id] = data
                self.dsm_cache[tile_id] = dsm
                self.label_cache[tile_id] = label

        h, w = data.shape[-2:]
        wh, ww = self.window_size
        x1 = random.randint(0, h - wh - 1)
        y1 = random.randint(0, w - ww - 1)
        x2, y2 = x1 + wh, y1 + ww

        data_p = data[:, x1:x2, y1:y2]
        dsm_p = dsm[x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]
        data_p, dsm_p, label_p = self._augment(data_p, dsm_p, label_p)

        return torch.from_numpy(data_p), torch.from_numpy(dsm_p), torch.from_numpy(label_p)


class TargetDataset(torch.utils.data.Dataset):
    def __init__(self, proto: Dict, window_size: Tuple[int, int], epoch_steps: int, cache=True):
        self.ids = proto["train_ids"]
        self.data_t = proto["data_t"]
        self.dsm_t = proto["dsm_t"]
        self.window_size = window_size
        self.epoch_steps = epoch_steps
        self.cache = cache
        self.data_cache = {}
        self.dsm_cache = {}

        for tile_id in self.ids:
            for p in [self.data_t.format(tile_id), self.dsm_t.format(tile_id)]:
                if not os.path.isfile(p):
                    raise FileNotFoundError(p)

    def __len__(self):
        return self.epoch_steps

    def _augment(self, img, dsm):
        if random.random() < 0.5:
            img = img[:, ::-1, :]
            dsm = dsm[::-1, :]
        if random.random() < 0.5:
            img = img[:, :, ::-1]
            dsm = dsm[:, ::-1]
        return np.copy(img), np.copy(dsm)

    def __getitem__(self, idx):
        tile_id = random.choice(self.ids)
        if tile_id in self.data_cache:
            data = self.data_cache[tile_id]
            dsm = self.dsm_cache[tile_id]
        else:
            img = io.imread(self.data_t.format(tile_id))
            if img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]
            data = (img.transpose((2, 0, 1)) / 255.0).astype("float32")

            dsm_raw = np.asarray(io.imread(self.dsm_t.format(tile_id)), dtype="float32")
            dmin, dmax = np.min(dsm_raw), np.max(dsm_raw)
            dsm = ((dsm_raw - dmin) / (dmax - dmin + 1e-8)).astype("float32")

            if self.cache:
                self.data_cache[tile_id] = data
                self.dsm_cache[tile_id] = dsm

        h, w = data.shape[-2:]
        wh, ww = self.window_size
        x1 = random.randint(0, h - wh - 1)
        y1 = random.randint(0, w - ww - 1)
        x2, y2 = x1 + wh, y1 + ww

        data_p = data[:, x1:x2, y1:y2]
        dsm_p = dsm[x1:x2, y1:y2]
        data_p, dsm_p = self._augment(data_p, dsm_p)

        return torch.from_numpy(data_p), torch.from_numpy(dsm_p)


def compute_metrics(predictions, gts, num_classes=6):
    cm = confusion_matrix(gts, predictions, labels=list(range(num_classes)))
    iou = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm) + 1e-8)
    miou = float(np.nanmean(iou[:5]))
    f1 = np.zeros(num_classes)
    for i in range(num_classes):
        denom = np.sum(cm[i, :]) + np.sum(cm[:, i])
        f1[i] = (2.0 * cm[i, i] / denom) if denom > 0 else 0.0
    mean_f1 = float(np.nanmean(f1[:5]))
    total = float(np.sum(cm))
    total_acc = float(np.trace(cm) / total * 100.0) if total > 0 else 0.0
    return {
        "mean_miou": miou,
        "mean_f1": mean_f1,
        "total_acc": total_acc,
    }


def evaluate_target(net, target_proto, window_size, stride, batch_size, device, max_tiles=0):
    test_ids = target_proto["test_ids"]
    if max_tiles and max_tiles > 0:
        test_ids = test_ids[:max_tiles]
    data_t = target_proto["data_t"]
    dsm_t = target_proto["dsm_t"]
    eroded_t = target_proto["eroded_t"]

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for tile_id in tqdm(test_ids, desc="Eval target", leave=False):
            img = io.imread(data_t.format(tile_id))
            if img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]
            img = (img / 255.0).astype("float32")

            dsm = np.asarray(io.imread(dsm_t.format(tile_id)), dtype="float32")
            dmin, dmax = np.min(dsm), np.max(dsm)
            dsm = (dsm - dmin) / (dmax - dmin + 1e-8)

            gt_rgb = np.asarray(io.imread(eroded_t.format(tile_id)))
            gt = convert_from_color(gt_rgb, ISPRS_PALETTE).astype("int64")

            pred_map = np.zeros(img.shape[:2] + (6,), dtype=np.float32)
            total = max(1, count_sliding_window(img, step=stride, window_size=window_size) // batch_size)
            for coords in tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False):
                img_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                img_t = torch.from_numpy(np.asarray(img_patches)).to(device)
                dsm_tens = torch.from_numpy(np.asarray(dsm_patches)).to(device)

                outs = net(img_t, dsm_tens, mode="Test")
                outs = outs.detach().cpu().numpy()
                for out, (x, y, w, h) in zip(outs, coords):
                    pred_map[x:x + w, y:y + h] += out.transpose((1, 2, 0))

            pred = np.argmax(pred_map, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt)

    preds = np.concatenate([p.ravel() for p in all_preds])
    gts = np.concatenate([g.ravel() for g in all_gts])
    return compute_metrics(preds, gts)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for week3 weak-cross training.")

    seed = int(os.environ.get("SSRS_SEED", "42"))
    set_seed(seed)

    data_root = os.environ.get("SSRS_DATA_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "autodl-tmp", "dataset")))
    source_dataset = os.environ.get("SSRS_SOURCE_DATASET", "Vaihingen")
    target_dataset = os.environ.get("SSRS_TARGET_DATASET", "Potsdam")
    mode = os.environ.get("SSRS_WEEK3_MODE", "uda-high").lower()  # source-only | uda-high

    if mode not in {"source-only", "uda-high"}:
        raise ValueError("SSRS_WEEK3_MODE must be one of: source-only, uda-high")

    window_size_int = int(os.environ.get("SSRS_WINDOW_SIZE", "256"))
    window_size = (window_size_int, window_size_int)
    batch_size = int(os.environ.get("SSRS_BATCH_SIZE", "8"))
    epoch_steps = int(os.environ.get("SSRS_EPOCH_STEPS", "400"))
    epochs = int(os.environ.get("SSRS_EPOCHS", "20"))
    num_workers = int(os.environ.get("SSRS_NUM_WORKERS", "4"))
    pin_memory = os.environ.get("SSRS_PIN_MEMORY", "1") == "1"
    persistent_workers = os.environ.get("SSRS_PERSISTENT_WORKERS", "0") == "1"
    prefetch_factor = int(os.environ.get("SSRS_PREFETCH_FACTOR", "2"))
    drop_last = os.environ.get("SSRS_DROP_LAST", "1") == "1"
    data_cache = os.environ.get("SSRS_DATA_CACHE", "0") == "1"
    base_lr = float(os.environ.get("SSRS_BASE_LR", "0.01"))
    adv_lambda = float(os.environ.get("SSRS_LAMBDA_ADV", "0.001"))
    adv_lr = float(os.environ.get("SSRS_ADV_LR", str(base_lr)))
    grl_lambda = float(os.environ.get("SSRS_GRL_LAMBDA", "1.0"))
    eval_stride = int(os.environ.get("SSRS_EVAL_STRIDE", "32"))
    eval_every = max(1, int(os.environ.get("SSRS_EVAL_EVERY", "1")))
    eval_max_tiles = int(os.environ.get("SSRS_EVAL_MAX_TILES", "0"))

    log_dir = os.environ.get("SSRS_LOG_DIR", "./runs/week3_weak_cross")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"MFNet_week3_{source_dataset}2{target_dataset}_seed{seed}.csv")

    src_proto = get_protocol(source_dataset, data_root)
    tgt_proto = get_protocol(target_dataset, data_root)

    cpu_count = os.cpu_count() or 8
    if num_workers < 0:
        num_workers = 0
    if num_workers > cpu_count:
        num_workers = cpu_count
    if num_workers == 0:
        persistent_workers = False
    if prefetch_factor < 1:
        prefetch_factor = 2

    print(f"[Week3] mode={mode}, source={source_dataset}, target={target_dataset}")
    print(f"[Week3] epochs={epochs}, steps={epoch_steps}, batch={batch_size}, lr={base_lr}, lambda_adv={adv_lambda}")
    print(
        f"[Week3] workers={num_workers}, pin_memory={pin_memory}, "
        f"persistent_workers={persistent_workers}, prefetch_factor={prefetch_factor}, drop_last={drop_last}"
    )
    print(f"[Week3] data_cache={data_cache} (warning: with workers>0, cache=True may consume large host RAM)")
    print(f"[Week3] eval_every={eval_every}, eval_stride={eval_stride}, eval_max_tiles={eval_max_tiles} (0=all)")

    src_set = SourceDataset(src_proto, window_size=window_size, epoch_steps=batch_size * epoch_steps, cache=data_cache)
    tgt_set = TargetDataset(tgt_proto, window_size=window_size, epoch_steps=batch_size * epoch_steps, cache=data_cache)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs.update({
            "persistent_workers": persistent_workers,
            "prefetch_factor": prefetch_factor,
        })

    src_loader = torch.utils.data.DataLoader(src_set, **loader_kwargs)
    tgt_loader = torch.utils.data.DataLoader(tgt_set, **loader_kwargs)

    device = torch.device("cuda")
    net = MFNet(num_classes=6).to(device)
    disc = DomainAdversarialHead(in_channels=256, hidden_channels=128, grl_lambda=grl_lambda).to(device)

    optimizer_seg = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
    optimizer_disc = optim.Adam(disc.parameters(), lr=adv_lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer_seg, [25, 35, 45], gamma=0.1)

    scaler = amp.GradScaler("cuda", enabled=os.environ.get("SSRS_MFNET_USE_AMP", "1") == "1")
    class_weights = torch.ones(6, device=device)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "iter", "mode", "source", "target", "train_seg_loss", "train_adv_feat_loss",
            "train_adv_disc_loss", "lambda_adv", "target_mean_miou", "target_mean_f1", "target_total_acc", "lr", "timestamp"
        ])

    iter_idx = 0
    best_miou = -1.0

    tgt_iter = iter(tgt_loader)

    for epoch in range(1, epochs + 1):
        net.train()
        disc.train()

        losses_seg = []
        losses_adv_feat = []
        losses_adv_disc = []

        for src_data, src_dsm, src_label in tqdm(src_loader, desc=f"Epoch {epoch}", leave=False):
            try:
                tgt_data, tgt_dsm = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_data, tgt_dsm = next(tgt_iter)

            src_data = Variable(src_data.to(device, non_blocking=True))
            src_dsm = Variable(src_dsm.to(device, non_blocking=True))
            src_label = Variable(src_label.to(device, non_blocking=True))
            tgt_data = Variable(tgt_data.to(device, non_blocking=True))
            tgt_dsm = Variable(tgt_dsm.to(device, non_blocking=True))

            optimizer_seg.zero_grad()
            with amp.autocast("cuda", enabled=scaler.is_enabled()):
                src_logits, src_feats = net(src_data, src_dsm, mode="Train", return_feat=True, feat_levels=("high",))
                loss_seg = cross_entropy_2d(src_logits, src_label, weight=class_weights)

                loss_adv_feat = src_logits.new_zeros(())
                if mode == "uda-high":
                    _, tgt_feats = net(tgt_data, tgt_dsm, mode="Train", return_feat=True, feat_levels=("high",))
                    src_dom_logits = disc(src_feats["high"], reverse=True)
                    tgt_dom_logits = disc(tgt_feats["high"], reverse=True)
                    loss_adv_feat = 0.5 * (
                        disc.domain_loss(src_dom_logits, 0.0) + disc.domain_loss(tgt_dom_logits, 1.0)
                    )

                total_loss = loss_seg + adv_lambda * loss_adv_feat

            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                scaler.step(optimizer_seg)
                scaler.update()
            else:
                total_loss.backward()
                optimizer_seg.step()

            loss_adv_disc = src_logits.new_zeros(())
            if mode == "uda-high":
                optimizer_disc.zero_grad()
                with torch.no_grad():
                    _, src_feats_det = net(src_data, src_dsm, mode="Train", return_feat=True, feat_levels=("high",))
                    _, tgt_feats_det = net(tgt_data, tgt_dsm, mode="Train", return_feat=True, feat_levels=("high",))

                src_dom_logits = disc(src_feats_det["high"].detach(), reverse=False)
                tgt_dom_logits = disc(tgt_feats_det["high"].detach(), reverse=False)
                loss_adv_disc = 0.5 * (
                    disc.domain_loss(src_dom_logits, 0.0) + disc.domain_loss(tgt_dom_logits, 1.0)
                )
                loss_adv_disc.backward()
                optimizer_disc.step()

            iter_idx += 1
            losses_seg.append(float(loss_seg.item()))
            losses_adv_feat.append(float(loss_adv_feat.item()))
            losses_adv_disc.append(float(loss_adv_disc.item()))

        do_eval = (epoch % eval_every == 0) or (epoch == epochs)
        if do_eval:
            net.eval()
            target_metrics = evaluate_target(
                net,
                tgt_proto,
                window_size=window_size,
                stride=eval_stride,
                batch_size=batch_size,
                device=device,
                max_tiles=eval_max_tiles,
            )
        else:
            target_metrics = {
                "mean_miou": float("nan"),
                "mean_f1": float("nan"),
                "total_acc": float("nan"),
            }

        mean_seg = float(np.mean(losses_seg)) if losses_seg else 0.0
        mean_adv_feat = float(np.mean(losses_adv_feat)) if losses_adv_feat else 0.0
        mean_adv_disc = float(np.mean(losses_adv_disc)) if losses_adv_disc else 0.0
        lr_now = float(optimizer_seg.param_groups[0]["lr"])

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                iter_idx,
                mode,
                source_dataset,
                target_dataset,
                mean_seg,
                mean_adv_feat,
                mean_adv_disc,
                adv_lambda,
                target_metrics["mean_miou"],
                target_metrics["mean_f1"],
                target_metrics["total_acc"],
                lr_now,
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ])

        if do_eval:
            print(
                f"[Week3][Epoch {epoch}] seg={mean_seg:.4f} adv_f={mean_adv_feat:.4f} "
                f"adv_d={mean_adv_disc:.4f} target_mIoU={target_metrics['mean_miou']:.4f}"
            )
        else:
            print(
                f"[Week3][Epoch {epoch}] seg={mean_seg:.4f} adv_f={mean_adv_feat:.4f} "
                f"adv_d={mean_adv_disc:.4f} target_mIoU=SKIP"
            )

        if do_eval and target_metrics["mean_miou"] > best_miou:
            best_miou = target_metrics["mean_miou"]
            torch.save(net.state_dict(), os.path.join(log_dir, "UNetformer_week3_best.pth"))

        torch.save(net.state_dict(), os.path.join(log_dir, "UNetformer_week3_last.pth"))
        scheduler.step()

    print(f"[Week3] Done. Best target mIoU={best_miou:.4f}")
    print(f"[Week3] CSV: {csv_path}")


if __name__ == "__main__":
    main()
