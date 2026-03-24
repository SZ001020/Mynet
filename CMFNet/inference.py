# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
import torch
from torch.autograd import Variable
import os
from Utils.utils import *
from Utils.CMFNet import *

# 配置路径和参数 (保持与 trainAndtest.py 一致)
WINDOW_SIZE = (256, 256)
STRIDE = 32
IN_CHANNELS = 3
BATCH_SIZE = 10
LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
N_CLASSES = len(LABELS)
DATASET = 'Vaihingen'
FOLDER = "/root/autodl-tmp/dataset/"

if DATASET == 'Potsdam':
    MAIN_FOLDER = FOLDER + 'Potsdam/'
    DATA_FOLDER = MAIN_FOLDER + '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'
    DSM_FOLDER = MAIN_FOLDER + '1_DSM_normalisation/dsm_potsdam_{}_normalized_lastools.jpg'
    LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
    ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'
    test_ids = ['2_13', '2_14', '3_13', '3_14', '4_13', '4_14', '4_15', '5_13', '5_14', '5_15', '6_13', '6_14', '6_15', '7_13']
elif DATASET == 'Vaihingen':
    MAIN_FOLDER = FOLDER + 'Vaihingen/'
    DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    DSM_FOLDER = MAIN_FOLDER + 'dsm/dsm_09cm_matching_area{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'
    test_ids = ['1', '3', '5', '7', '11', '13', '15', '17', '21', '23', '26', '28', '30', '32', '34', '37']

# 初始化网络
net = CMFNet(in_channels=IN_CHANNELS, out_channels=N_CLASSES)
net.cuda()

# 加载训练好的权重
model_path = '/root/SSRS/CMFNet/segnet_final'
if os.path.exists(model_path):
    print(f"Loading weights from {model_path}...")
    net.load_state_dict(torch.load(model_path))
else:
    print(f"Error: {model_path} not found!")
    exit()

def run_inference():
    net.eval()
    output_dir = '/root/SSRS/CMFNet/predictions'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Starting inference on {len(test_ids)} images...")
    
    # 这里的 test 函数定义在 trainAndtest.py 中，由于我们在同一目录下，
    # 我们可以直接重新定义一个简化的推理逻辑或从 trainAndtest 拷贝
    with torch.no_grad():
        for id in test_ids:
            print(f"Processing Image {id}...")
            img = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32'))
            dsm = np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32')
            
            # 使用滑动窗口进行推理
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))
            
            # 这里调用滑动窗口逻辑
            # 为了简单起见，我们直接调用之前 trainAndtest.py 里的逻辑
            # (由于 test 函数逻辑较长，这里直接运行简易版)
            
            # 建立坐标
            window_size = WINDOW_SIZE
            stride = STRIDE
            batch_size = BATCH_SIZE
            
            coords = list(sliding_window(img, step=stride, window_size=window_size))
            
            for i, chunk in enumerate(grouper(batch_size, coords)):
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in chunk]
                image_patches = torch.from_numpy(np.asarray(image_patches)).cuda()
                
                # DSM 归一化
                dsm_slice = dsm # 简化处理
                dsm_min, dsm_max = np.min(dsm), np.max(dsm)
                dsm_normalized = (dsm - dsm_min) / (dsm_max - dsm_min)
                
                dsm_patches = [np.copy(dsm_normalized[x:x + w, y:y + h]) for x, y, w, h in chunk]
                dsm_patches = torch.from_numpy(np.asarray(dsm_patches)).cuda()
                
                outs = net(image_patches, dsm_patches)
                outs = outs.data.cpu().numpy()
                
                for out, (x, y, w, h) in zip(outs, chunk):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
            
            pred = np.argmax(pred, axis=-1)
            
            # 转换为彩色并保存
            color_pred = convert_to_color(pred)
            save_path = os.path.join(output_dir, f'prediction_{id}.png')
            io.imsave(save_path, color_pred)
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    run_inference()
