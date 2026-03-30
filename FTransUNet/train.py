import numpy as np
import os
import csv
from glob import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import random
import time
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from torch.autograd import Variable
# from IPython.display import clear_output
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
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
CSV_LOG_PATH = os.path.join(LOG_DIR, "FTransUNet_{}_seed{}.csv".format(DATASET, SEED))
CSV_STANDARD_FIELDS = [
    "model", "dataset", "seed", "epoch", "iter", "train_loss",
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

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))
print("Device :", nvmlDeviceGetName(handle))

config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = 6
config_vit.n_skip = 3
config_vit.patches.grid = (int(256 / 16), int(256 / 16))
net = ViT_seg(config_vit, img_size=256, num_classes=6).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path))
params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)
# Load the datasets

print("training : ", train_ids)
print("testing : ", test_ids)
print("BATCH_SIZE: ", BATCH_SIZE)
print("Stride Size: ", Stride_Size)
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
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params':[value],'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params':[value],'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, return_details=False):
    # Use the network on the test set
    ## Potsdam
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in test_ids)
        # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids)
    ## Vaihingen
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
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
                dsm = (dsm - min) / (max - min)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = torch.from_numpy(dsm_patches).cuda(non_blocking=True)

                # Do the inference
                outs = net(image_patches, dsm_patches)
                outs = outs.detach().cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            # clear_output()
            
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


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    # criterion = nn.NLLLoss(weight=weights)
    iter_ = 0
    acc_best = 90.0
    save_best = os.environ.get("SSRS_SAVE_BEST", "1") == "1"
    save_last = os.environ.get("SSRS_SAVE_LAST", "1") == "1"
    save_interval = int(os.environ.get("SSRS_SAVE_INTERVAL", "0"))
    # Keep checkpoints with the same directory used for CSV logs.
    save_dir = LOG_DIR
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, 'FTransUNet_best.pth')
    last_ckpt_path = os.path.join(save_dir, 'FTransUNet_last.pth')
    eval_every = int(os.environ.get("SSRS_FTRANS_EVAL_EVERY", str(len(train_loader))))
    print("[FTransUNet] Eval every {} iterations".format(eval_every))

    for e in range(1, epochs + 1):
        net.train()
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = Variable(data.cuda()), Variable(dsm.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data, dsm)
            loss = CrossEntropy2d(output, target, weight=weights)
            loss.backward()
            optimizer.step()

            losses[iter_] = float(loss.item())
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            if iter_ % 100 == 0:
                # clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)

            # if e % save_epoch == 0:
            if eval_every > 0 and iter_ % eval_every == 0:
                net.eval()
                acc, metric_details = test(net, test_ids, all=False, stride=eval_stride, return_details=True)
                net.train()
                current_lr = optimizer.param_groups[0]["lr"]
                append_csv_logger(CSV_LOG_PATH, {
                    "model": "FTransUNet",
                    "dataset": DATASET,
                    "seed": SEED,
                    "epoch": e,
                    "iter": iter_,
                    "train_loss": float(mean_losses[max(0, iter_ - 1)]),
                    "val_metric": float(acc),
                    "best_val_metric": float(max(acc_best, acc)),
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
                if acc > acc_best:
                    if save_best:
                        torch.save(net.state_dict(), best_ckpt_path)
                    acc_best = acc

        if save_interval > 0 and e % save_interval == 0:
            interval_ckpt = os.path.join(save_dir, 'FTransUNet_epoch{}_acc{:.4f}.pth'.format(e, acc_best))
            torch.save(net.state_dict(), interval_ckpt)

        if save_last:
            torch.save(net.state_dict(), last_ckpt_path)

        if scheduler is not None:
            scheduler.step()

    print('acc_best: ', acc_best)

if __name__ == "__main__":
    #####   train   ####
    time_start=time.time()
    epochs_run = int(os.environ.get("SSRS_EPOCHS", "50"))
    save_epoch_run = int(os.environ.get("SSRS_SAVE_EPOCH", "1"))
    train(net, optimizer, epochs_run, scheduler, save_epoch=save_epoch_run)
    time_end=time.time()
    print('Total Time Cost: ',time_end-time_start)

    #####   test   ####
    # 加载刚才训练中最好的模型（或者指定一个已有模型）
    # best_model_path = './results_final/...' 
    net.eval()
    print("Generating prediction maps...")
    acc, all_preds, all_gts = test(net, test_ids, all=True, stride=Stride_Size)
    print("Final Test Acc: ", acc)
    for p, id_ in zip(all_preds, test_ids):
        img = convert_to_color(p)
        if not os.path.exists('./results_prediction/'):
            os.makedirs('./results_prediction/')
        io.imsave('./results_prediction/inference_tile{}.png'.format(id_), img)
    print("Predictions saved to ./results_prediction/")
