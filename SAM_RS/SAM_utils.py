import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import skimage
from skimage.segmentation import find_boundaries
from PIL import Image
import time

device ='cuda'
sam = sam_model_registry ["vit_h"] (checkpoint ="./sam_vit_h_4b8939.pth")
sam.to( device = device )

mask_generator = SamAutomaticMaskGenerator(sam, crop_nms_thresh=0.5, box_nms_thresh=0.5, pred_iou_thresh=0.96)

def SAMAug(tI , mask_generator):
    masks = mask_generator.generate(tI)
    if len(masks) == 0:
        return
    tI= skimage.img_as_float (tI)

    BoundaryPrior = np.zeros (( tI. shape [0] , tI. shape [1]))
    BoundaryPrior_output = np.zeros ((tI.shape [0] , tI. shape [1]))
    
    Objects_first_few =  np.zeros (( tI. shape [0] , tI. shape [1]))
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    idx=1
    for ann in sorted_anns:        
        if ann['area'] < 50:
            continue
        if idx==51:
            break
        m = ann['segmentation']
        color_mask = idx
        print(color_mask)
        Objects_first_few[m] = color_mask
        idx=idx+1

    for maskindex in range(len(masks)):
        thismask =masks[ maskindex ][ 'segmentation']
        mask_=np.zeros (( thismask.shape ))
        mask_[np.where( thismask == True)]=1
        BoundaryPrior = BoundaryPrior + find_boundaries (mask_ ,mode='thick')

    BoundaryPrior [np.where( BoundaryPrior >0) ]=1
    BoundaryPrior_index=np.where(BoundaryPrior >0)
    Objects_first_few[BoundaryPrior_index]= 0  
    BoundaryPrior_output [np.where( BoundaryPrior >0) ]=255
    BoundaryPrior_output = BoundaryPrior_output.astype(np.uint8) 
    return BoundaryPrior_output,Objects_first_few  

directory_name = '/root/autodl-tmp/dataset/Vaihingen/top/'

img_list=[f for f in os.listdir(directory_name)]
img_list=sorted(img_list)
start_time=time.time()
for img_input in img_list:
    if img_input.endswith('.png'):
        img_name=img_input.split(".")[0]
        image_type=".png"
        image = Image.open(directory_name+img_input)
        image = np.array(image)
        print(type(image))
        print(image.shape)
        BoundaryPrior_output, Objects_first_few=SAMAug(image, mask_generator)
        image_boundary = Image.fromarray(BoundaryPrior_output)
        Objects_first_few = Objects_first_few.astype(np.uint8)
        image_objects = Image.fromarray(Objects_first_few)
        image_boundary.save("./SAM/LoveDA_obj_data/"+img_name+'_Boundary'+image_type)
        image_objects.save("./SAM/LoveDA_obj_data/"+img_name+'_objects'+image_type)
end_time = time.time()
run_time = end_time - start_time
print(f"Runing time: {run_time} second.")

# --- 修改后的部分 ---

# 1. 输入路径：你的 Vaihingen 原始图片目录
directory_name = '/root/autodl-tmp/dataset/Vaihingen/top/' 

# 2. 输出路径：创建对应的存储目录
boundary_out = '/root/autodl-tmp/dataset/Vaihingen/sam_boundary_merge/'
object_out = '/root/autodl-tmp/dataset/Vaihingen/V_merge/'

os.makedirs(boundary_out, exist_ok=True)
os.makedirs(object_out, exist_ok=True)

img_list = [f for f in os.listdir(directory_name) if f.endswith('.tif')]
img_list = sorted(img_list)

start_time = time.time()
for img_input in img_list:
    # 提取 ID，例如从 top_mosaic_09cm_area1.tif 提取出 1
    # 这样保存的文件名就能匹配 utils.py 的 ISPRS_merge_{}.tif 格式
    try:
        img_id = img_input.split('area')[-1].split('.')[0]
    except:
        continue

    print(f"Processing ID: {img_id} ...")
    
    image = Image.open(os.path.join(directory_name, img_input))
    image = np.array(image)
    
    # 生成 Boundary 和 Object 图
    BoundaryPrior_output, Objects_first_few = SAMAug(image, mask_generator)
    
    # 保存结果（文件名必须严格按照 utils.py 要求的格式）
    image_boundary = Image.fromarray(BoundaryPrior_output)
    image_boundary.save(os.path.join(boundary_out, f'ISPRS_merge_{img_id}.tif'))
    
    Objects_first_few = Objects_first_few.astype(np.uint8)
    image_objects = Image.fromarray(Objects_first_few)
    image_objects.save(os.path.join(object_out, f'V_merge_{img_id}.tif'))

# --- 修改结束 ---