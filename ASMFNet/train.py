import numpy as np
from skimage import io
from glob import glob
from tqdm.auto import tqdm
import random
import itertools
import matplotlib
matplotlib.use('agg')
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
import os
import csv
from models.swinfusenet.vision_transformer import SwinFuseNet as ViT_seg
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener
    
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

DATASET_NAME = os.environ.get("SSRS_DATASET", "Vaihingen")
LOG_DIR = os.environ.get("SSRS_LOG_DIR", "./runs/week1_baseline")
os.makedirs(LOG_DIR, exist_ok=True)
CSV_LOG_PATH = os.path.join(LOG_DIR, "ASMFNet_{}_seed{}.csv".format(DATASET_NAME, SEED))
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

# Parameters
# Parameters
WINDOW_SIZE_INT = int(os.environ.get("SSRS_ASMF_WINDOW_SIZE", os.environ.get("SSRS_WINDOW_SIZE", "224")))
if WINDOW_SIZE_INT % 28 != 0:
    print("[ASMFNet] WARNING: window size {} is incompatible with Swin window=7 (requires multiple of 28). Fallback to 224.".format(WINDOW_SIZE_INT))
    WINDOW_SIZE_INT = 224
WINDOW_SIZE = (WINDOW_SIZE_INT, WINDOW_SIZE_INT) # Patch size
STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
# FOLDER = "D:/RA/CUHK-S/root/autodl-tmp/dataset/ISPRS_semantic_labeling_Vaihingen/"
DATA_ROOT = os.environ.get("SSRS_DATA_ROOT", "/root/SSRS/autodl-tmp/dataset")
if not DATA_ROOT.endswith('/'):
    DATA_ROOT += '/'
FOLDER = os.environ.get("ASMFNET_DATA_ROOT", DATA_ROOT + "Vaihingen/")
if not FOLDER.endswith('/'):
    FOLDER += '/'
BATCH_SIZE = int(os.environ.get("SSRS_BATCH_SIZE", "10")) # Number of samples in a mini-batch

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memorycd x
# DATASET = 'Vaihingen'
# MAIN_FOLDER = FOLDER + 'Vaihingen/'
MAIN_FOLDER = FOLDER
DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
DSM_FOLDER = MAIN_FOLDER + 'dsm/dsm_09cm_matching_area{}.tif'
LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'

class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        self.dsm_files = [DSM_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.dsm_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        self.dsm_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        # Default epoch size is 10 000 samples
        return 10000

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = io.imread(self.data_files[random_idx])

            data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.dsm_cache_.keys():
            dsm = self.dsm_cache_[random_idx]
        else:
            # DSM is normalized in [0, 1]
            dsm = np.asarray(io.imread(self.dsm_files[random_idx]), dtype='float32')

            min = np.min(dsm)
            max = np.max(dsm)
            dsm = (dsm - min) / (max - min)
            if self.cache:
                self.dsm_cache_[random_idx] = dsm

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        dsm_p = dsm[x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        data_p, dsm_p, label_p = self.data_augmentation(data_p, dsm_p, label_p)
        # print((torch.from_numpy(dsm_p).shape))
        # print((torch.from_numpy(data_p).shape))
        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(dsm_p),
                torch.from_numpy(label_p))
        
load_path= './pretrain/swin_tiny_patch4_window7_224.pth'
# load_path= '/content/drive/My Drive/SwinFuseNet/pretrain/swin_tiny_patch4_window7_224.pth'
net = ViT_seg(img_size=WINDOW_SIZE_INT, num_classes=6).to(device)
print('Model img_size: {}x{}'.format(WINDOW_SIZE_INT, WINDOW_SIZE_INT))
params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print('Params: ', params)
net.load_from(load_path)
# Load the datasets
#net = nn.DataParallel(net)
train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
test_ids = ['5', '21', '15', '30']

print('Tiles for training :', train_ids)
print('Tiles for testing :', test_ids)

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

base_lr = float(os.environ.get("SSRS_BASE_LR", "0.001"))
eval_stride = int(os.environ.get("SSRS_EVAL_STRIDE", "32"))
eval_num_tiles = int(os.environ.get("SSRS_EVAL_NUM_TILES", "0"))
eval_every_epochs = int(os.environ.get("SSRS_ASMF_EVAL_EVERY", "1"))
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


def test(net, test_ids, num1, num2, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, return_details=False):
    # Use the network on the test set
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (convert_from_color(io.imread(LABEL_FOLDER.format(id))) for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []
    count = 0
    with torch.no_grad():
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids), leave=False):
            # e1 = 0
            # len1 = len(list(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))))

            pred = np.zeros(img.shape[:2] + (N_CLASSES,))
            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = torch.from_numpy(image_patches).to(device, non_blocking=True)

                min = np.min(dsm)
                max = np.max(dsm)
                dsm = (dsm - min) / (max - min)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = torch.from_numpy(dsm_patches).to(device, non_blocking=True)

                # Do the inference
                with amp.autocast("cuda", enabled=os.environ.get("SSRS_ASMF_USE_AMP", "1") == "1"):
                    outs = net(image_patches, dsm_patches)
                outs = outs.detach().cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            
            all_preds.append(pred)
            # all_gts.append(gt)
            all_gts.append(gt_e)
            # Compute some metrics
            # metrics(pred.ravel(), gt_e.ravel())
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


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=2):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.to(device)

    # criterion = nn.NLLLoss2d(weight=weights)
    iter_ = 0
    acc_best = 10
    oa = 0
    save_best = os.environ.get("SSRS_SAVE_BEST", "1") == "1"
    save_last = os.environ.get("SSRS_SAVE_LAST", "1") == "1"
    save_interval = int(os.environ.get("SSRS_SAVE_INTERVAL", "0"))
    save_dir = os.environ.get("SSRS_ASMF_SAVE_DIR", "/root/SSRS/ASMFNet/res2")
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "ASMFNet_best.pth")
    last_ckpt_path = os.path.join(save_dir, "ASMFNet_last.pth")
    use_amp = os.environ.get("SSRS_ASMF_USE_AMP", "1") == "1"
    scaler = amp.GradScaler("cuda", enabled=use_amp)
    print("[ASMFNet] Eval every {} epoch(s), eval tiles: {} (0 means all)".format(eval_every_epochs, eval_num_tiles))

    for e in range(1, epochs + 1):
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            net.train()
            data = data.to(device, non_blocking=True)
            dsm = dsm.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()
            with amp.autocast("cuda", enabled=use_amp):
                output = net(data, dsm)
                loss = CrossEntropy2d(output, target, weight=weights, reduction='mean')
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            losses[iter_] = float(loss.item())
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            if iter_ % 100 == 0:
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                oa = accuracy(pred, gt)
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, oa))
                
            iter_ += 1

            del (data, target, loss)

        if e % eval_every_epochs == 0:
            eval_test_ids = test_ids[:eval_num_tiles] if eval_num_tiles > 0 else test_ids
            acc, metric_details = test(net, eval_test_ids, all=False, stride=eval_stride, num1=e, num2=batch_idx, return_details=True)
            current_lr = optimizer.param_groups[0]["lr"]
            append_csv_logger(CSV_LOG_PATH, {
                "model": "ASMFNet",
                "dataset": DATASET_NAME,
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
            oa = 0
            if acc > acc_best:
                if save_best:
                    print('Saving best model to {}...'.format(best_ckpt_path))
                    torch.save(net.state_dict(), best_ckpt_path)
                acc_best = acc

            if save_interval > 0 and e % save_interval == 0:
                interval_ckpt = os.path.join(save_dir, 'ASMFNet_epoch{}_acc{:.4f}.pth'.format(e, acc))
                print('Saving interval model to {}...'.format(interval_ckpt))
                torch.save(net.state_dict(), interval_ckpt)

        if save_last:
            torch.save(net.state_dict(), last_ckpt_path)

        if scheduler is not None:
            scheduler.step()
    print("Train Done!!")

epochs_run = int(os.environ.get("SSRS_EPOCHS", "100"))
save_epoch_run = int(os.environ.get("SSRS_SAVE_EPOCH", "2"))
train(net, optimizer, epochs_run, scheduler, save_epoch=save_epoch_run)

# ######   test   ####
# acc, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
# print("Acc: ", acc)
# for p, id_ in zip(all_preds, test_ids):
#     img = convert_to_color(p)
#     # plt.imshow(img) and plt.show()
#     io.imsave('./inference9064_tile_{}.png'.format(id_), img)
