import numpy as np
from skimage import io
import torch
from utils import *
from models.swinfusenet.vision_transformer import SwinFuseNet as ViT_seg
import os
from tqdm import tqdm

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = '/root/SSRS/ASMFNet/res2/segnet256_epoch24_999_82.1615.pth'
OUTPUT_DIR = '/root/SSRS/ASMFNet/inference_results/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据集路径及类别
DATA_ROOT = os.environ.get("SSRS_DATA_ROOT", "/root/SSRS/autodl-tmp/dataset")
if not DATA_ROOT.endswith('/'):
    DATA_ROOT += '/'
FOLDER = os.environ.get("ASMFNET_DATA_ROOT", DATA_ROOT + "Vaihingen/")
if not FOLDER.endswith('/'):
    FOLDER += '/'
DATA_FOLDER = FOLDER + 'top/top_mosaic_09cm_area{}.tif'
DSM_FOLDER = FOLDER + 'dsm/dsm_09cm_matching_area{}.tif'
LABEL_FOLDER = FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
test_ids = ['5', '21', '15', '30']
WINDOW_SIZE = (224, 224)
BATCH_SIZE = 10
N_CLASSES = 6

STRIDE_SMOOTH = 64  # 使用 64 的步长进行平滑推理

def run_smooth_inference():
    # 1. 加载模型
    print(f"Loading model from {BEST_MODEL_PATH}...")
    net = ViT_seg(num_classes=N_CLASSES).to(device)
    net.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    net.eval()

    # 2. 准备推理数据推理逻辑 (参考 train.py 中的 test 函数)
    with torch.no_grad():
        for id_ in test_ids:
            print(f"Processing tile {id_}...")
            # 读取图片
            img = 1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id_)), dtype='float32')
            dsm = np.asarray(io.imread(DSM_FOLDER.format(id_)), dtype='float32')
            
            # DSM 归一化
            dsm_min, dsm_max = np.min(dsm), np.max(dsm)
            dsm = (dsm - dsm_min) / (dsm_max - dsm_min)
            
            # 初始化预测背景
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))
            
            # 关键：减小步长
            coords = list(sliding_window(img, step=STRIDE_SMOOTH, window_size=WINDOW_SIZE))
            
            # 按 Batch 分组
            for i in tqdm(range(0, len(coords), BATCH_SIZE)):
                batch_coords = coords[i:i + BATCH_SIZE]
                
                # 提取 Patch
                image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2, 0, 1)) for x, y, w, h in batch_coords]
                dsm_patches = [np.copy(dsm[x:x+w, y:y+h]) for x, y, w, h in batch_coords]
                
                image_patches = torch.from_numpy(np.asarray(image_patches)).to(device)
                dsm_patches = torch.from_numpy(np.asarray(dsm_patches)).to(device)
                
                # 模型预测
                outs = net(image_patches, dsm_patches)
                outs = outs.data.cpu().numpy()
                
                # 填充预测结果
                for out, (x, y, w, h) in zip(outs, batch_coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x+w, y:y+h] += out
            
            # 取通道最大值得出最终分类
            pred_labels = np.argmax(pred, axis=-1)
            
            # 3. 转换为颜色图并保存
            color_img = convert_to_color(pred_labels)
            save_path = os.path.join(OUTPUT_DIR, f'inference_tile_{id_}.png')
            io.imsave(save_path, color_img)
            print(f"Saved inference results to {save_path}")

if __name__ == "__main__":
    run_smooth_inference()
