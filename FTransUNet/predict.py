import torch
import os
import numpy as np
from skimage import io
from train import net, test, test_ids, Stride_Size
from utils import convert_to_color

# 1. 定义最佳权重路径
BEST_MODEL_PATH = './resultsv_se_ablation/segnet256_epoch25_92.2374690110549'
SAVE_DIR = './results_prediction/'

def main():
    # 创建保存目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    # 2. 加载权重
    print(f"Loading weights from: {BEST_MODEL_PATH}")
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Model file {BEST_MODEL_PATH} not found!")
        return
        
    net.load_state_dict(torch.load(BEST_MODEL_PATH))
    net.eval()

    # 3. 运行推理
    print("Generating prediction maps, please wait...")
    acc, all_preds, all_gts = test(net, test_ids, all=True, stride=Stride_Size)
    
    print(f"Test Accuracy: {acc:.2f}%")

    # 4. 保存为彩色图像
    for p, id_ in zip(all_preds, test_ids):
        img_colored = convert_to_color(p)
        save_path = os.path.join(SAVE_DIR, f'inference_tile{id_}.png')
        io.imsave(save_path, img_colored)
        print(f"Saved prediction: {save_path}")

if __name__ == "__main__":
    main()
