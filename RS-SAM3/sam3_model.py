"""
SAM3 模型封装 — 适配遥感 per-class 二分类推理。
基于 Medical-SAM3 的 SAM3Model 类设计。
"""

import sys, os
import numpy as np
import torch
from PIL import Image

SE = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, SE)
os.chdir(SE)

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3Model:
    """SAM3 二分类推理封装."""

    def __init__(self, confidence_threshold=0.1, device='cuda', checkpoint_path=None):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.processor = None

    def load_model(self):
        if self.model is not None:
            return
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        bpe_path = f'{SE}/sam3/assets/bpe_simple_vocab_16e6.txt.gz'

        if self.checkpoint_path:
            self.model = build_sam3_image_model(
                bpe_path=bpe_path, checkpoint_path=None, load_from_HF=False, device=self.device)
            ckpt = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
            sd = ckpt.get('model', ckpt)
            if any('detector.' in k for k in sd):
                sd = {k.replace('detector.', ''): v for k, v in sd.items() if 'detector' in k}
            self.model.load_state_dict(sd, strict=False)
            print(f"Loaded custom checkpoint: {self.checkpoint_path}")
        else:
            self.model = build_sam3_image_model(
                bpe_path=bpe_path,
                checkpoint_path=f'{SE}/weights/sam3/sam3.pt',
                device=self.device)

        self.processor = Sam3Processor(self.model, confidence_threshold=self.confidence_threshold, device=self.device)
        print("SAM3 model loaded.")

    @torch.no_grad()
    def encode_image(self, image):
        self.load_model()
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.processor.set_image(image)

    @torch.no_grad()
    def predict_binary(self, inference_state, text_prompt):
        """
        Return binary mask (H,W) numpy for a text prompt.
        Combines instance masks (clean binary) + semantic head (top-k thresholded).
        """
        self.processor.reset_all_prompts(inference_state)
        state = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        h, w = state['original_height'], state['original_width']
        combined = torch.zeros((h, w), dtype=torch.bool, device=self.device)

        # 1. Instance masks (clean binary from transformer decoder)
        masks = state.get('masks')
        if masks is not None and len(masks) > 0:
            for m in masks[:10]:
                combined = combined | m.bool()

        # 2. Semantic head: top-15% activation as positive
        sem = state.get('semantic_mask_logits')
        if sem is not None:
            from torch.nn.functional import interpolate
            sem_prob = sem.squeeze().sigmoid()
            if sem_prob.dim() > 2:
                sem_prob = sem_prob.squeeze()
            k = max(int(sem_prob.numel() * 0.15), 100)
            threshold = max(sem_prob.flatten().topk(k).values[-1].item(), 0.70)
            sem_mask = sem_prob > threshold
            if sem_mask.shape != (h, w):
                sem_mask = interpolate(sem_mask.float().unsqueeze(0).unsqueeze(0),
                                       (h, w), mode='nearest').squeeze() > 0.5
            combined = combined | sem_mask

        if combined.sum() < 10:
            return np.zeros((h, w), dtype=np.uint8)
        return combined.cpu().numpy().astype(np.uint8)

    def cleanup(self):
        del self.model; del self.processor; self.model = None; self.processor = None
        import gc; gc.collect(); torch.cuda.empty_cache()
