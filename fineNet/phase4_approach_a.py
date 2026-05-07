"""
Phase 4-A: DSM 高程先验注入 — 基于 Phase 1 验证的 mmseg 评估框架

修改: 在 SegEarthOV3Segmentation 的 predict() 中，
加载 DSM 数据作为 logit bias，提升 ground/object 分类的精度。
"""

import os, sys, json, gc, time
from datetime import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

PROJECT_ROOT = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, '/root/Mynet/sam3-main')
os.chdir(PROJECT_ROOT)

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.logging import MMLogger
from collections import OrderedDict
from prettytable import PrettyTable
from mmseg.evaluation.metrics import IoUMetric
from mmseg.registry import METRICS
from mmengine.logging import print_log

import custom_datasets  # noqa
import segearthov3_segmentor  # noqa


# ============================================================
# Custom evaluator (same as Phase 1)
# ============================================================
@METRICS.register_module()
class DetailedIoUMetric(IoUMetric):
    def compute_metrics(self, results: list) -> dict:
        import numpy as np
        results = list(zip(*results))
        t = [sum(r) for r in results[:4]]
        rm = self.total_area_to_metrics(t[0],t[1],t[2],t[3],self.metrics,self.nan_to_num,self.beta)
        s = OrderedDict({k:np.round(np.nanmean(v)*100,2) for k,v in rm.items()})
        m = {}
        for k,v in s.items(): m['m'+k if k!='aAcc' else k] = v
        rm.pop('aAcc',None)
        for mk,mv in rm.items():
            rv = np.round(mv*100,2)
            for i,n in enumerate(self.dataset_meta['classes']): m[f'{mk}.{n}']=rv[i]
        return m


# ============================================================
# DSM Prior injection
# ============================================================

def compute_ndsm(dsm, ground_kernel_size=101):
    from scipy.ndimage import grey_opening
    ground = grey_opening(dsm, size=ground_kernel_size)
    return dsm - ground


def load_dsm_prior(image_path, dataset_name):
    """Load DSM and compute normalized prior [0,1]."""
    if dataset_name == 'vaihingen':
        tile = os.path.basename(image_path).replace('.tif', '')
        tile = tile.replace('top_mosaic_09cm_', 'dsm_09cm_matching_')
        dsm_path = os.path.join('/root/autodl-tmp/dataset/Vaihingen/dsm', f'{tile}.tif')
    elif dataset_name == 'potsdam':
        tile = os.path.basename(image_path).replace('_RGB.tif', '')
        tile = tile.replace('top_potsdam_', 'dsm_potsdam_')
        dsm_path = os.path.join('/root/autodl-tmp/dataset/Potsdam/1_DSM', f'{tile}.tif')
    else:
        return None

    if not os.path.exists(dsm_path):
        return None

    dsm = np.array(Image.open(dsm_path))
    ndsm = compute_ndsm(dsm)
    # Normalize: height 0m → 0, height 10m → 1
    return np.clip(ndsm / 10.0, 0, 1).astype(np.float32)


# ============================================================
# Monkey-patch: inject DSM prior into the segmentor
# ============================================================

def create_dsm_predict(dataset_name):
    """包装 predict 方法，在推理时注入 DSM prior logit bias."""
    from segearthov3_segmentor import SegEarthOV3Segmentation

    original_predict = SegEarthOV3Segmentation.predict

    def dsm_predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [ds.metainfo for ds in data_samples]
        else:
            batch_img_metas = [dict(ori_shape=inputs.shape[2:])] * inputs.shape[0]

        for i, meta in enumerate(batch_img_metas):
            image_path = meta.get('img_path')
            image = Image.open(image_path).convert('RGB')

            # Pre-resize
            max_edge = 2000
            if max(image.size) > max_edge:
                ratio = max_edge / max(image.size)
                image = image.resize(
                    (int(image.size[0]*ratio), int(image.size[1]*ratio)),
                    Image.BILINEAR)
            ori_shape = meta['ori_shape']

            # === Load DSM prior ===
            dsm_prior = load_dsm_prior(image_path, dataset_name)

            # === Modified inference with DSM ===
            w, h = image.size
            seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)

            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                inference_state = self.processor.set_image(image)

                for query_idx, query_word in enumerate(self.query_words):
                    self.processor.reset_all_prompts(inference_state)
                    inference_state = self.processor.set_text_prompt(
                        state=inference_state, prompt=query_word)

                    # Instance masks
                    if self.use_transformer_decoder and \
                       inference_state.get('masks_logits') is not None:
                        inst_logits = inference_state['masks_logits']
                        if inst_logits.shape[0] > 0:
                            for inst_id in range(min(inst_logits.shape[0], 10)):
                                il = inst_logits[inst_id].squeeze()
                                score = inference_state.get('object_score')
                                s = score[inst_id] if score is not None and score.numel() > inst_id else 1.0
                                if il.shape != (h, w):
                                    il = F.interpolate(il.view(1,1,*il.shape),
                                        size=(h,w), mode='bilinear', align_corners=False).squeeze()
                                seg_logits[query_idx] = torch.max(seg_logits[query_idx], il * s)

                    # Semantic mask
                    if self.use_sem_seg:
                        sm = inference_state.get('semantic_mask_logits')
                        if sm is not None:
                            if sm.dim() == 2: sm = sm.unsqueeze(0).unsqueeze(0)
                            elif sm.dim() == 3: sm = sm.unsqueeze(0)
                            if sm.shape[-2:] != (h,w):
                                sm = F.interpolate(sm, size=(h,w), mode='bilinear', align_corners=False)
                            seg_logits[query_idx] = torch.max(seg_logits[query_idx], sm.squeeze())

                    # === DSM prior ===
                    if dsm_prior is not None:
                        dsm_t = torch.from_numpy(dsm_prior).float().to(self.device)
                        if dsm_t.shape != (h, w):
                            dsm_t = F.interpolate(dsm_t.view(1,1,*dsm_t.shape),
                                size=(h,w), mode='bilinear', align_corners=False).squeeze()
                        # Object classes (building, tree, car): boost in high areas
                        if query_idx in [1, 3, 4]:
                            seg_logits[query_idx] *= (1.0 + dsm_t * 3.0)
                        # Ground classes (road, grass): dampen in high areas
                        if query_idx in [0, 2]:
                            seg_logits[query_idx] *= 1.0 / (1.0 + dsm_t * 5.0)

                    # Presence score
                    ps = inference_state.get('presence_score')
                    if ps is not None and isinstance(ps, torch.Tensor):
                        seg_logits[query_idx] *= ps.squeeze()

            # Post-processing
            if self.num_cls != self.num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = torch.nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(self.num_cls, len(self.query_idx), 1, 1)
                seg_logits = (seg_logits * cls_index.to(self.device)).max(1)[0]

            if seg_logits.shape[-2:] != ori_shape:
                seg_logits = F.interpolate(seg_logits.unsqueeze(0),
                    size=ori_shape, mode='bilinear', align_corners=False).squeeze(0)

            seg_pred = torch.argmax(seg_logits, dim=0)
            max_vals = seg_logits.max(0)[0]
            seg_pred[max_vals < self.prob_thd] = self.bg_idx

            from mmengine.structures import PixelData
            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': seg_logits}),
                'pred_sem_seg': PixelData(**{'data': seg_pred.unsqueeze(0)})
            })

        return data_samples

    return dsm_predict


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['vaihingen', 'potsdam', 'both'], default='both')
    parser.add_argument('--output-dir', default='/root/Mynet/autodl-tmp/runs')
    args = parser.parse_args()

    datasets = ['vaihingen', 'potsdam'] if args.dataset == 'both' else [args.dataset]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output_dir, f'phase4_dsm_A_{timestamp}')
    os.makedirs(output_root, exist_ok=True)
    logger = MMLogger.get_instance('phase4a', log_file=os.path.join(output_root,'experiment.log'),
                                    file_mode='w', log_level='INFO')

    # Config maps (no-clutter)
    configs = {
        'vaihingen': './configs/cfg_vaihingen_noclutter.py',
        'potsdam': './configs/cfg_potsdam_noclutter.py',
    }

    all_results = []

    for ds_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Phase 4-A: DSM Prior — {ds_name}")

        # === 1. Baseline (no DSM) ===
        logger.info("Running baseline (no DSM)...")
        cfg = Config.fromfile(configs[ds_name])
        cfg.model.use_transformer_decoder = True
        cfg.model.use_sem_seg = True
        cfg.model.use_presence_score = False
        cfg.test_evaluator = dict(type='DetailedIoUMetric', iou_metrics=['mIoU'])
        cfg.work_dir = os.path.join(output_root, f'{ds_name}_baseline')
        os.makedirs(cfg.work_dir, exist_ok=True)

        runner = Runner.from_cfg(cfg)
        results_base = runner.test()
        del runner; gc.collect(); torch.cuda.empty_cache()

        base_5c = compute_5c_miou(results_base)
        logger.info(f"  Baseline: 6c-mIoU={results_base.get('mIoU',0):.1f}%, 5c-mIoU={base_5c:.1f}%")
        results_base['dataset'] = ds_name
        results_base['config'] = 'Baseline (no DSM)'
        results_base['5c_mIoU'] = base_5c
        all_results.append(results_base)

        # === 2. DSM-Enhanced ===
        logger.info("Running DSM-enhanced...")
        from segearthov3_segmentor import SegEarthOV3Segmentation
        original_predict = SegEarthOV3Segmentation.predict
        SegEarthOV3Segmentation.predict = create_dsm_predict(ds_name)

        cfg2 = Config.fromfile(configs[ds_name])
        cfg2.model.use_transformer_decoder = True
        cfg2.model.use_sem_seg = True
        cfg2.model.use_presence_score = False
        cfg2.test_evaluator = dict(type='DetailedIoUMetric', iou_metrics=['mIoU'])
        cfg2.work_dir = os.path.join(output_root, f'{ds_name}_dsm')
        os.makedirs(cfg2.work_dir, exist_ok=True)

        runner2 = Runner.from_cfg(cfg2)
        results_dsm = runner2.test()
        del runner2; gc.collect(); torch.cuda.empty_cache()

        # Restore original
        SegEarthOV3Segmentation.predict = original_predict

        dsm_5c = compute_5c_miou(results_dsm)
        delta = dsm_5c - base_5c
        logger.info(f"  DSM-A: 6c-mIoU={results_dsm.get('mIoU',0):.1f}%, 5c-mIoU={dsm_5c:.1f}%")
        logger.info(f"  Δ vs baseline: {delta:+.1f}%")
        results_dsm['dataset'] = ds_name
        results_dsm['config'] = 'DSM-Enhanced (logit bias)'
        results_dsm['5c_mIoU'] = dsm_5c
        all_results.append(results_dsm)

    # === Summary ===
    logger.info(f"\n{'='*60}")
    logger.info("Phase 4-A Summary:")
    for ds_name in datasets:
        bs = [r for r in all_results if r.get('dataset') == ds_name and 'Baseline' in r.get('config','')]
        ds = [r for r in all_results if r.get('dataset') == ds_name and 'DSM' in r.get('config','')]
        if bs and ds:
            delta = ds[0]['5c_mIoU'] - bs[0]['5c_mIoU']
            logger.info(f"  {ds_name}: baseline={bs[0]['5c_mIoU']:.1f}% → DSM={ds[0]['5c_mIoU']:.1f}% (Δ={delta:+.1f}%)")

    # Save
    json.dump(all_results, open(os.path.join(output_root, 'results.json'), 'w'), indent=2, default=str)
    logger.info(f"\nResults: {output_root}/results.json")


def compute_5c_miou(metrics):
    iou_keys = [k for k in metrics if k.startswith('IoU.') and 'clutter' not in k.lower()]
    vals = [metrics[k] for k in iou_keys]
    return sum(vals)/len(vals) if vals else 0.0


if __name__ == '__main__':
    main()
