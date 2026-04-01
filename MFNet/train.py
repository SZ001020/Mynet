import numpy as np
import os
import csv
import random
from glob import glob
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
import time
import cv2
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch import amp
from utils import *
from torch.autograd import Variable
from IPython.display import clear_output
from UNetFormer_MMSAM import UNetFormer as MFNet
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = int(os.environ.get("SSRS_SEED", "42"))
set_seed(SEED)

PERF_MODE = os.environ.get("SSRS_PERF_MODE", "1") == "1"
if PERF_MODE:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

LOG_DIR = os.environ.get("SSRS_LOG_DIR", "./runs/week1_baseline")
os.makedirs(LOG_DIR, exist_ok=True)
CSV_LOG_PATH = os.path.join(LOG_DIR, "MFNet_{}_seed{}.csv".format(DATASET, SEED))
CSV_STANDARD_FIELDS = [
    "model", "dataset", "seed", "epoch", "iter", "train_loss",
    "train_loss_ce", "train_loss_boundary", "train_loss_object", "loss_mode",
    "lambda_bdy", "lambda_obj", "structure_supervision",
    "warmup_scale", "conf_threshold", "conf_kept_ratio",
    "val_metric", "best_val_metric", "lr", "batch_size", "window_size", "stride",
    "total_acc", "mean_f1", "kappa", "mean_miou",
    "roads_f1", "buildings_f1", "low_veg_f1", "trees_f1", "cars_f1", "clutter_f1",
    "timestamp",
]


def init_csv_logger(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_STANDARD_FIELDS)


def append_csv_logger(csv_path, row_dict):
    ordered_row = [row_dict.get(k, "") for k in CSV_STANDARD_FIELDS]
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ordered_row)


init_csv_logger(CSV_LOG_PATH)

os.makedirs('./resultsv', exist_ok=True)
os.makedirs('./resultsp', exist_ok=True)
os.makedirs('./resultsh', exist_ok=True)

net = MFNet(num_classes=N_CLASSES).cuda()

params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print('All Params:   ', params)

params1 = 0
params2 = 0
for name, param in net.image_encoder.named_parameters():
    # if "Adapter" not in name:
    if "lora_" not in name:
    # if "lora_" not in name and "Adapter" not in name:
        params1 += param.nelement()
    else:
        params2 += param.nelement()
print('ImgEncoder:   ', params1)
# print('Adapter:       ', params2)
print('Lora: ', params2)
# print('Adapter_Lora: ', params2)
print('Others: ', params-params1-params2)

# for name, parms in net.named_parameters():
#     print('%-50s' % name, '%-30s' % str(parms.shape), '%-10s' % str(parms.nelement()))

# params = 0
# for name, param in net.sam.prompt_encoder.named_parameters():
#     params += param.nelement()
# print('prompt_encoder: ', params)

# params = 0
# for name, param in net.sam.mask_decoder.named_parameters():
#     params += param.nelement()
# print('mask_decoder: ', params)

# print(net)

print("training : ", len(train_ids))
print("testing : ", len(test_ids))
train_set = ISPRS_dataset(train_ids, cache=CACHE)
num_workers = int(os.environ.get("SSRS_NUM_WORKERS", "16"))
pin_memory = os.environ.get("SSRS_PIN_MEMORY", "1") == "1"
persistent_workers = os.environ.get("SSRS_PERSISTENT_WORKERS", "1") == "1"
prefetch_factor = int(os.environ.get("SSRS_PREFETCH_FACTOR", "4"))

loader_kwargs = {
    "batch_size": BATCH_SIZE,
    "shuffle": True,
}
if num_workers > 0:
    loader_kwargs.update({
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
    })

train_loader = torch.utils.data.DataLoader(train_set, **loader_kwargs)

base_lr = float(os.environ.get("SSRS_BASE_LR", "0.01"))
eval_stride = int(os.environ.get("SSRS_EVAL_STRIDE", str(Stride_Size)))
optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [12, 17, 22], gamma=0.1)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 7, 9], gamma=0.1)

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, return_details=False):
    # Use the network on the test set
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
        # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids)
    ## Vaihingen
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    if DATASET == 'Hunan':
        eroded_labels = ((np.asarray(io.imread(ERODED_FOLDER.format(id)), dtype='int64')) for id in test_ids)
    else:
        eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)

    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    with torch.no_grad():
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids), leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = torch.from_numpy(image_patches).cuda(non_blocking=True)

                min = np.min(dsm)
                max = np.max(dsm)
                if DATASET == 'Hunan':
                    dsm = (dsm - min) / (max - min + 1e-8)
                else:
                    dsm = (dsm - min) / (max - min)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = torch.from_numpy(dsm_patches).cuda(non_blocking=True)

                # Do the inference
                outs = net(image_patches, dsm_patches, mode='Test')
                outs = outs.detach().cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()
    
    if DATASET == 'Hunan':
        metric_result = metrics_loveda(
            np.concatenate([p.ravel() for p in all_preds]),
            np.concatenate([p.ravel() for p in all_gts]).ravel(),
            return_details=return_details,
        )
    else:
        metric_result = metrics(
            np.concatenate([p.ravel() for p in all_preds]),
            np.concatenate([p.ravel() for p in all_gts]).ravel(),
            return_details=return_details,
        )

    if return_details:
        accuracy, metric_details = metric_result
    else:
        accuracy = metric_result

    if all:
        if return_details:
            return accuracy, all_preds, all_gts, metric_details
        return accuracy, all_preds, all_gts
    else:
        if return_details:
            return accuracy, metric_details
        return accuracy


def run_prior_quality_check():
    if os.environ.get("SSRS_PRIOR_QUALITY_CHECK", "1") != "1":
        return
    if os.environ.get("SSRS_USE_STRUCTURE_LOSS", "1") != "1":
        return

    sample_limit = int(os.environ.get("SSRS_PRIOR_CHECK_SAMPLES", "0"))
    strict_mode = os.environ.get("SSRS_PRIOR_QUALITY_STRICT", "0") == "1"

    ids = train_ids[:sample_limit] if sample_limit > 0 else train_ids
    if len(ids) == 0 or not BOUNDARY_FOLDER or not OBJECT_FOLDER:
        print("[PriorCheck] Skip: no train ids or no prior templates configured.")
        return

    missing_boundary = []
    missing_object = []
    bdy_ratios = []
    obj_ratios = []
    obj_counts = []

    for tile_id in ids:
        bdy_path = BOUNDARY_FOLDER.format(tile_id)
        obj_path = OBJECT_FOLDER.format(tile_id)

        if not os.path.isfile(bdy_path):
            missing_boundary.append(bdy_path)
        if not os.path.isfile(obj_path):
            missing_object.append(obj_path)
        if (not os.path.isfile(bdy_path)) or (not os.path.isfile(obj_path)):
            continue

        bdy = np.asarray(io.imread(bdy_path))
        if bdy.ndim == 3:
            bdy = bdy[:, :, 0]
        bdy_bin = (bdy > 0).astype(np.uint8)

        obj = np.asarray(io.imread(obj_path))
        if obj.ndim == 3:
            obj = obj[:, :, 0]
        obj = obj.astype(np.int64)

        bdy_ratios.append(float(np.mean(bdy_bin)))
        obj_ratios.append(float(np.mean(obj > 0)))
        obj_counts.append(int(len(np.unique(obj[obj > 0]))))

    print("[PriorCheck] tiles={} missing_bdy={} missing_obj={}".format(len(ids), len(missing_boundary), len(missing_object)))
    if missing_boundary:
        print("[PriorCheck] missing boundary examples: {}".format(missing_boundary[:3]))
    if missing_object:
        print("[PriorCheck] missing object examples: {}".format(missing_object[:3]))

    if bdy_ratios:
        bdy_mean = float(np.mean(bdy_ratios))
        bdy_min = float(np.min(bdy_ratios))
        bdy_max = float(np.max(bdy_ratios))
        obj_mean = float(np.mean(obj_ratios))
        obj_min = float(np.min(obj_ratios))
        obj_max = float(np.max(obj_ratios))
        obj_cnt_mean = float(np.mean(obj_counts))

        print("[PriorCheck] boundary positive ratio mean/min/max = {:.6f}/{:.6f}/{:.6f}".format(bdy_mean, bdy_min, bdy_max))
        print("[PriorCheck] object positive ratio   mean/min/max = {:.6f}/{:.6f}/{:.6f}".format(obj_mean, obj_min, obj_max))
        print("[PriorCheck] object instances per tile mean = {:.2f}".format(obj_cnt_mean))

        suspicious = []
        if bdy_mean < 0.003:
            suspicious.append("boundary ratio too low")
        if obj_mean < 0.02:
            suspicious.append("object ratio too low")
        if obj_cnt_mean < 5:
            suspicious.append("object instance count too low")

        if suspicious:
            msg = "[PriorCheck][WARN] Potential low-quality priors: {}".format(", ".join(suspicious))
            if strict_mode:
                raise RuntimeError(msg)
            print(msg)
    elif strict_mode:
        raise RuntimeError("[PriorCheck] No valid prior files found under strict mode.")


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()
    use_amp = os.environ.get("SSRS_MFNET_USE_AMP", "1") == "1"
    micro_bs = int(os.environ.get("SSRS_MFNET_MICRO_BS", "2"))
    eval_num_tiles = int(os.environ.get("SSRS_EVAL_NUM_TILES", "0"))
    loss_mode = os.environ.get("SSRS_LOSS_MODE", "SEG+BDY+OBJ")
    lambda_bdy = float(os.environ.get("SSRS_LAMBDA_BDY", "0.1"))
    lambda_obj = float(os.environ.get("SSRS_LAMBDA_OBJ", "1.0"))
    use_structure = os.environ.get("SSRS_USE_STRUCTURE_LOSS", "1") == "1"
    warmup_epochs = int(os.environ.get("SSRS_STRUCTURE_WARMUP_EPOCHS", "10"))
    conf_threshold = float(os.environ.get("SSRS_STRUCTURE_CONF_THRESH", "0.6"))
    scaler = amp.GradScaler("cuda", enabled=use_amp)
    criterion_b = BoundaryLoss()
    criterion_o = ObjectLoss()

    valid_loss_modes = {"SEG", "SEG+BDY", "SEG+OBJ", "SEG+BDY+OBJ"}
    if loss_mode not in valid_loss_modes:
        raise ValueError("Unsupported SSRS_LOSS_MODE='{}', expected one of {}".format(loss_mode, sorted(valid_loss_modes)))

    iter_ = 0
    MIoU_best = 0.00
    save_best = os.environ.get("SSRS_SAVE_BEST", "1") == "1"
    save_last = os.environ.get("SSRS_SAVE_LAST", "1") == "1"
    save_interval = int(os.environ.get("SSRS_SAVE_INTERVAL", "0"))

    # Keep checkpoints with the same directory used for CSV logs.
    save_dir = LOG_DIR
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, '{}_best.pth'.format(MODEL))
    last_ckpt_path = os.path.join(save_dir, '{}_last.pth'.format(MODEL))

    run_prior_quality_check()

    for e in range(1, epochs + 1):
        if warmup_epochs > 0:
            warmup_scale = min(1.0, float(max(0, e - 1)) / float(warmup_epochs))
        else:
            warmup_scale = 1.0
        lambda_bdy_eff = lambda_bdy * warmup_scale
        lambda_obj_eff = lambda_obj * warmup_scale

        net.train()
        start_time = time.time()
        conf_kept_ratio_epoch = []
        for batch_idx, (data, dsm, boundary, object_map, target) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_size_curr = data.shape[0]
            step_micro_bs = max(1, min(micro_bs, batch_size_curr))
            num_chunks = (batch_size_curr + step_micro_bs - 1) // step_micro_bs
            batch_loss = 0.0
            batch_loss_ce = 0.0
            batch_loss_boundary = 0.0
            batch_loss_object = 0.0
            output_vis = None
            target_vis = None

            for start in range(0, batch_size_curr, step_micro_bs):
                end = min(start + step_micro_bs, batch_size_curr)
                data_chunk = Variable(data[start:end].cuda(non_blocking=True))
                dsm_chunk = Variable(dsm[start:end].cuda(non_blocking=True))
                boundary_chunk = Variable(boundary[start:end].cuda(non_blocking=True))
                object_chunk = Variable(object_map[start:end].cuda(non_blocking=True))
                target_chunk = Variable(target[start:end].cuda(non_blocking=True))

                with amp.autocast("cuda", enabled=use_amp):
                    output = net(data_chunk, dsm_chunk, mode='Train')
                    loss_ce = loss_calc(output, target_chunk, weights)

                    if use_structure and conf_threshold > 0.0:
                        conf_map = torch.softmax(output.detach(), dim=1).amax(dim=1)
                        conf_mask = (conf_map >= conf_threshold).float()
                    else:
                        conf_mask = torch.ones_like(boundary_chunk, dtype=output.dtype)

                    conf_kept_ratio_epoch.append(float(conf_mask.mean().item()))

                    boundary_gated = boundary_chunk.float() * conf_mask
                    object_gated = torch.where(
                        conf_mask > 0.5,
                        object_chunk,
                        torch.zeros_like(object_chunk),
                    )

                    if use_structure and loss_mode in ("SEG+BDY", "SEG+BDY+OBJ"):
                        loss_boundary = criterion_b(output, boundary_gated)
                    else:
                        loss_boundary = output.new_zeros(())

                    if use_structure and loss_mode in ("SEG+OBJ", "SEG+BDY+OBJ"):
                        loss_object = criterion_o(output, object_gated)
                    else:
                        loss_object = output.new_zeros(())

                    if loss_mode == "SEG":
                        loss_full = loss_ce
                    elif loss_mode == "SEG+BDY":
                        loss_full = loss_ce + lambda_bdy_eff * loss_boundary
                    elif loss_mode == "SEG+OBJ":
                        loss_full = loss_ce + lambda_obj_eff * loss_object
                    else:
                        loss_full = loss_ce + lambda_bdy_eff * loss_boundary + lambda_obj_eff * loss_object

                    loss = loss_full / num_chunks

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                batch_loss += float(loss_full.item())
                batch_loss_ce += float(loss_ce.item())
                batch_loss_boundary += float(loss_boundary.item())
                batch_loss_object += float(loss_object.item())
                if output_vis is None:
                    output_vis = output.detach()
                    target_vis = target_chunk.detach()

                del data_chunk, dsm_chunk, boundary_chunk, object_chunk, target_chunk, output, loss_ce, loss_boundary, loss_object, loss_full, loss

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            losses[iter_] = batch_loss
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 99):iter_ + 1])

            if iter_ % 100 == 0:
                clear_output()
                pred = np.argmax(output_vis.cpu().numpy()[0], axis=0)
                gt = target_vis.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), batch_loss, accuracy(pred, gt)))
            iter_ += 1

            del (output_vis, target_vis)

        if e % save_epoch == 0:
            train_time = time.time()
            print("Training time: {:.3f} seconds".format(train_time - start_time))
            # We validate with the largest possible stride for faster computing
            eval_test_ids = test_ids[:eval_num_tiles] if eval_num_tiles > 0 else test_ids
            net.eval()
            MIoU, metric_details = test(net, eval_test_ids, all=False, stride=eval_stride, return_details=True)
            net.train()
            test_time = time.time()
            print("Test time: {:.3f} seconds".format(test_time - train_time))
            current_lr = optimizer.param_groups[0]["lr"]
            append_csv_logger(CSV_LOG_PATH, {
                "model": "MFNet",
                "dataset": DATASET,
                "seed": SEED,
                "epoch": e,
                "iter": iter_,
                "train_loss": float(mean_losses[max(0, iter_ - 1)]),
                "train_loss_ce": float(batch_loss_ce),
                "train_loss_boundary": float(batch_loss_boundary),
                "train_loss_object": float(batch_loss_object),
                "loss_mode": loss_mode,
                "lambda_bdy": float(lambda_bdy_eff),
                "lambda_obj": float(lambda_obj_eff),
                "structure_supervision": int(use_structure),
                "warmup_scale": float(warmup_scale),
                "conf_threshold": float(conf_threshold),
                "conf_kept_ratio": float(np.mean(conf_kept_ratio_epoch)) if conf_kept_ratio_epoch else 1.0,
                "val_metric": float(MIoU),
                "best_val_metric": float(max(MIoU_best, MIoU)),
                "lr": float(current_lr),
                "batch_size": BATCH_SIZE,
                "window_size": "{}x{}".format(WINDOW_SIZE[0], WINDOW_SIZE[1]),
                "stride": eval_stride,
                "total_acc": metric_details.get("total_acc"),
                "mean_f1": metric_details.get("mean_f1"),
                "kappa": metric_details.get("kappa"),
                "mean_miou": metric_details.get("mean_miou"),
                "roads_f1": metric_details.get("roads_f1"),
                "buildings_f1": metric_details.get("buildings_f1"),
                "low_veg_f1": metric_details.get("low_veg_f1"),
                "trees_f1": metric_details.get("trees_f1"),
                "cars_f1": metric_details.get("cars_f1"),
                "clutter_f1": metric_details.get("clutter_f1"),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            if MIoU > MIoU_best:
                if save_best:
                    torch.save(net.state_dict(), best_ckpt_path)
                MIoU_best = MIoU

            if save_interval > 0 and e % save_interval == 0:
                interval_ckpt = os.path.join(save_dir, '{}_epoch{}_miou{:.4f}.pth'.format(MODEL, e, MIoU))
                torch.save(net.state_dict(), interval_ckpt)

            torch.cuda.empty_cache()

        if save_last:
            torch.save(net.state_dict(), last_ckpt_path)

        if scheduler is not None:
            scheduler.step()
    print('MIoU_best: ', MIoU_best)

if MODE == 'Train':
    epochs_run = int(os.environ.get("SSRS_EPOCHS", str(epochs)))
    save_epoch_run = int(os.environ.get("SSRS_SAVE_EPOCH", str(save_epoch)))
    train(net, optimizer, epochs_run, scheduler, weights=WEIGHTS, save_epoch=save_epoch_run)

elif MODE == 'Test':
    if DATASET == 'Vaihingen':
        net.load_state_dict(torch.load('./resultsv/YOUR_MODEL'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsv/inference_UNetFormer_{}_tile_{}.png'.format('huge', id_), img)

    elif DATASET == 'Potsdam':
        net.load_state_dict(torch.load('./resultsp/YOUR_MODEL'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsp/inference_UNetFormer_{}_tile_{}.png'.format('huge', id_), img)

    elif DATASET == 'Hunan':
        net.load_state_dict(torch.load('./resultsh/YOUR_MODEL'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=128)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsh/inference_UNetFormer_{}_tile_{}.png'.format('base', id_), img)