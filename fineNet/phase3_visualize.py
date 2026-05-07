#!/usr/bin/env python3
"""Phase 3 推理图生成: 对比 Zero-shot vs Fine-tuned"""
import sys,os,torch,numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0,'/root/Mynet/SegEarth-OV-3-main')
sys.path.insert(0,'/root/Mynet/sam3-main')
sys.path.insert(0,'/root/Mynet/fineNet')

from finetune_model import FineTunedSAM3

CKPT_PATH = '/root/Mynet/autodl-tmp/runs/phase3_partial_20260429_231705/best_model_vaihingen.pt'
OUT_DIR = '/root/Mynet/autodl-tmp/runs/phase3_partial_20260429_231705'
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = [[255,255,255],[0,0,255],[0,255,255],[0,255,0],[255,255,0],[255,0,0]]
CLASSES = ['road','building','grass','tree','car','clutter']

SAMPLES = [
    ('vaihingen', '/root/Mynet/autodl-tmp/dataset/Vaihingen/top/top_mosaic_09cm_area1.tif',
     '/root/Mynet/autodl-tmp/dataset/Vaihingen/gts_index/top_mosaic_09cm_area1.png'),
    ('vaihingen', '/root/Mynet/autodl-tmp/dataset/Vaihingen/top/top_mosaic_09cm_area15.tif',
     '/root/Mynet/autodl-tmp/dataset/Vaihingen/gts_index/top_mosaic_09cm_area15.png'),
    ('potsdam', '/root/Mynet/autodl-tmp/dataset/Potsdam/2_Ortho_RGB/top_potsdam_2_10_RGB.tif',
     '/root/Mynet/autodl-tmp/dataset/Potsdam/labels_index/top_potsdam_2_10.png'),
]

def pred_to_color(pred, palette):
    h,w=pred.shape; c=np.zeros((h,w,3),dtype=np.uint8)
    for i,color in enumerate(palette): c[pred==i]=color
    return c

def gt_to_color(gt, palette):
    gt=np.clip(gt.astype(np.int32)-1,0,len(palette)-1)
    return pred_to_color(gt,palette)

print("Loading fine-tuned model...")
model = FineTunedSAM3(num_classes=5, mode='partial', device='cuda')
ckpt = torch.load(CKPT_PATH, map_location='cuda', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded: epoch={ckpt['epoch']}, V-mIoU={ckpt['v_miou']:.1f}%, P-mIoU={ckpt['p_miou']:.1f}%")

for ds_name, img_path, gt_path in SAMPLES:
    tile = os.path.splitext(os.path.basename(img_path))[0].replace('_RGB','')
    print(f"\n{ds_name}/{tile}")

    rgb = np.array(Image.open(img_path).convert('RGB'))
    gt = np.array(Image.open(gt_path)) if os.path.exists(gt_path) else None

    # Fine-tuned inference
    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
        img_tensor = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float().cuda()/255.0
        logits = model(img_tensor)
        pred_ft = logits[0].argmax(0).cpu().numpy()

    # Resize GT to match if needed
    if gt is not None and gt.shape != pred_ft.shape:
        gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize(
            (pred_ft.shape[1], pred_ft.shape[0]), Image.NEAREST))

    pred_ft_color = pred_to_color(pred_ft, PALETTE[:5]+PALETTE[5:])  # 5 classes only
    gt_color = gt_to_color(gt, PALETTE) if gt is not None else None

    # Save individual
    Image.fromarray(pred_ft_color).save(os.path.join(OUT_DIR, f'{tile}_finetuned.png'))

    # Comparison figure: RGB | GT | Zero-shot (load from phase1) | Fine-tuned
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    max_disp = 500; ratio = max_disp/max(rgb.shape[:2])
    rgb_disp = np.array(Image.fromarray(rgb).resize(
        (int(rgb.shape[1]*ratio), int(rgb.shape[0]*ratio)), Image.LANCZOS))
    pred_disp = np.array(Image.fromarray(pred_ft_color).resize(
        (int(pred_ft_color.shape[1]*ratio), int(pred_ft_color.shape[0]*ratio)), Image.NEAREST))

    axes[0].imshow(rgb_disp); axes[0].set_title('RGB', fontweight='bold'); axes[0].axis('off')

    if gt_color is not None:
        gt_disp = np.array(Image.fromarray(gt_color).resize(
            (int(gt_color.shape[1]*ratio), int(gt_color.shape[0]*ratio)), Image.NEAREST))
        axes[1].imshow(gt_disp); axes[1].set_title('Ground Truth', fontweight='bold')
    axes[1].axis('off')

    # Load zero-shot prediction from phase1_visualizations
    zs_path = f'/root/Mynet/autodl-tmp/runs/phase1_visualizations/{ds_name}_{tile}_Semantic-Only.png'
    zs_alt = f'/root/Mynet/autodl-tmp/runs/phase1_visualizations/{ds_name}_{tile}_Semantic.png'
    if os.path.exists(zs_path):
        zs = np.array(Image.open(zs_path))
    elif os.path.exists(zs_alt):
        zs = np.array(Image.open(zs_alt))
    else:
        zs = np.zeros_like(pred_ft_color)
    zs_disp = np.array(Image.fromarray(zs).resize(
        (int(zs.shape[1]*ratio), int(zs.shape[0]*ratio)), Image.NEAREST))
    axes[2].imshow(zs_disp); axes[2].set_title('Zero-shot (65.8%)', fontweight='bold'); axes[2].axis('off')

    axes[3].imshow(pred_disp); axes[3].set_title('Fine-tuned (52.1%)', fontweight='bold'); axes[3].axis('off')

    legend_patches = [mpatches.Patch(color=np.array(c)/255, label=n) for c,n in zip(PALETTE,CLASSES)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=6, fontsize=9, frameon=False)
    fig.suptitle(f'{ds_name.upper()} — Zero-shot vs Fine-tuned (Partial)', fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0,0.06,1,0.95])
    fig_path = os.path.join(OUT_DIR, f'comparison_zs_vs_ft_{tile}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig_path}")

del model; torch.cuda.empty_cache()
print("\nDone!")
