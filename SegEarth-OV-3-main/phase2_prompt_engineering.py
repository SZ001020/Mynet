#!/usr/bin/env python3
"""
Phase 2: Prompt Engineering 系统研究

对每个数据集运行 B/C/D 三组 prompt，每组仅测试 Semantic-Only 和 Dual-Head。
A 组（baseline）复用 Phase 1 的 no-clutter 结果。
输出 5 类 mIoU，所有实验均无 clutter。
"""

import os
import sys
import json
import time
import csv
import gc
import argparse
from datetime import datetime

PROJECT_ROOT = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
import numpy as np
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
# Custom Evaluator (same as Phase 1 — returns per-class metrics)
# ============================================================
@METRICS.register_module()
class DetailedIoUMetric(IoUMetric):
    def compute_metrics(self, results: list) -> dict:
        import numpy as np
        results = list(zip(*results))
        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)
        class_names = self.dataset_meta['classes']

        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        ret_metrics.pop('aAcc', None)
        for metric_name, metric_arr in ret_metrics.items():
            rounded = np.round(metric_arr * 100, 2)
            for cls_idx, cls_name in enumerate(class_names):
                metrics[f'{metric_name}.{cls_name}'] = rounded[cls_idx]

        ret_metrics_class = OrderedDict({
            metric_name: np.round(metric_value * 100, 2)
            for metric_name, metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)
        print_log('per class results:', logger='current')
        print_log('\n' + class_table_data.get_string(), logger='current')
        return metrics


# ============================================================
# 实验配置
# ============================================================
# Prompt 变体: (label, prompt_suffix, classname_file_pattern)
PROMPT_VARIANTS = {
    'A': {'label': 'A-Baseline', 'suffix': 'noclutter'},
    'B': {'label': 'B-RS-View', 'suffix': 'prompt_b'},
    'C': {'label': 'C-Geometry', 'suffix': 'prompt_c'},
    'D': {'label': 'D-Synonyms', 'suffix': 'prompt_d'},
}

# 仅保留最优的 2 个 head 配置
HEAD_CONFIGS = [
    ('Semantic-Only',     False, True,  False),
    ('Dual-Head',         True,  True,  False),
]

DATASETS = {
    'vaihingen': {'dir': 'Vaihingen'},
    'potsdam': {'dir': 'Potsdam'},
}

# Phase 1 no-clutter baseline results (reused for Group A)
PHASE1_NC_DIR = '/root/Mynet/autodl-tmp/runs/phase1_baseline_20260429_180550'


def _to_native(val):
    import numpy as np
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {k: _to_native(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_native(v) for v in val]
    return val


def extract_metrics(results):
    import numpy as np
    metrics = {}
    metrics['aAcc'] = _to_native(results.get('aAcc', 0))
    metrics['mIoU'] = _to_native(results.get('mIoU', 0))
    metrics['mAcc'] = _to_native(results.get('mAcc', 0))
    for key, val in results.items():
        if not isinstance(key, str) or key in ('aAcc', 'mIoU', 'mAcc'):
            continue
        if isinstance(val, (int, float, np.integer, np.floating, np.ndarray)):
            v = _to_native(val)
            if isinstance(v, (int, float)) and -0.01 < v < 101.0:
                metrics[key] = v
    return metrics


def compute_5c_miou(metrics):
    """Compute 5-class mIoU (excluding clutter=0)."""
    iou_keys = [k for k in metrics if k.startswith('IoU.') and 'clutter' not in k.lower()]
    if not iou_keys:
        return None
    values = [metrics[k] for k in iou_keys]
    return sum(values) / len(values)


def load_baseline_results(dataset_prefix):
    """从 Phase 1 结果中加载 Group A baseline."""
    results = {}
    nc_dir = PHASE1_NC_DIR
    for head_label, use_td, use_ss, use_ps in HEAD_CONFIGS:
        # 查找对应的 metrics.json
        exp_name = f'{dataset_prefix}_{head_label.replace(" ", "_").replace("+", "-")}'
        # actual folder name from Phase 1 uses underscores in dataset key
        exp_dir = os.path.join(nc_dir, f'{dataset_prefix}_nc{exp_name[len(dataset_prefix):]}')
        json_path = os.path.join(exp_dir, 'metrics.json')
        if not os.path.exists(json_path):
            # Try alternative naming
            for d in os.listdir(nc_dir):
                dpath = os.path.join(nc_dir, d)
                if d.startswith(dataset_prefix) and head_label.replace(' ', '_').replace('+', '-') in d:
                    json_path = os.path.join(dpath, 'metrics.json')
                    if os.path.exists(json_path):
                        break
        if os.path.exists(json_path):
            with open(json_path) as f:
                m = json.load(f)
            m['prompt_group'] = 'A-Baseline'
            results[head_label] = m
    return results


def run_single_experiment(dataset_key, head_label, use_td, use_ss,
                          config_path, output_dir, prompt_group, logger):
    """运行单个实验"""
    import torch as _t
    exp_name = f'{dataset_key}_{head_label.replace(" ", "_").replace("+", "-")}'
    work_dir = os.path.join(output_dir, exp_name)
    os.makedirs(work_dir, exist_ok=True)

    logger.info(f"  [{prompt_group}] {dataset_key}/{head_label}")

    start_time = time.time()
    runner = None
    try:
        cfg = Config.fromfile(config_path)
        cfg.model.use_transformer_decoder = use_td
        cfg.model.use_sem_seg = use_ss
        cfg.model.use_presence_score = False
        cfg.test_evaluator = dict(type='DetailedIoUMetric', iou_metrics=['mIoU'])
        cfg.work_dir = work_dir

        runner = Runner.from_cfg(cfg)
        results = runner.test()
        elapsed = time.time() - start_time

        metrics = extract_metrics(results)
        metrics['dataset'] = dataset_key
        metrics['config_label'] = head_label
        metrics['prompt_group'] = prompt_group
        metrics['use_transformer_decoder'] = use_td
        metrics['use_sem_seg'] = use_ss
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['elapsed_seconds'] = round(elapsed, 1)

        miou_5c = compute_5c_miou(metrics)
        logger.info(f"    6c-mIoU={metrics['mIoU']:.2f}  5c-mIoU={miou_5c:.2f}  ({elapsed:.0f}s)")

        detail_path = os.path.join(work_dir, 'metrics.json')
        with open(detail_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        return metrics
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"    FAILED ({elapsed:.0f}s): {e}")
        import traceback; traceback.print_exc()
        return None
    finally:
        if runner is not None:
            del runner
        gc.collect()
        _t.cuda.empty_cache()


def save_results_csv(csv_path, all_rows):
    if not all_rows:
        return
    all_keys = set()
    for row in all_rows:
        all_keys.update(row.keys())
    meta_keys = ['dataset', 'config_label', 'prompt_group']
    priority = ['aAcc', 'mIoU', 'mAcc']
    other_keys = sorted([k for k in all_keys if k not in meta_keys and k not in priority])
    metric_keys = [k for k in priority if k in other_keys] + [k for k in other_keys if k not in priority]
    fieldnames = meta_keys + metric_keys

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def main():
    args = parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output_root, f'phase2_prompt_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    log_path = os.path.join(output_root, 'experiment.log')
    logger = MMLogger.get_instance('phase2', log_file=log_path, file_mode='w', log_level='INFO')
    logger.info(f"Phase 2: Prompt Engineering (5-class, no clutter)")
    logger.info(f"Output: {output_root}")

    all_results = []

    for dataset_key in args.datasets:
        ds_info = DATASETS[dataset_key]
        logger.info(f"\n{'#'*60}")
        logger.info(f"Dataset: {dataset_key}")

        # Group A: load from Phase 1
        logger.info(f"  Loading baseline (Group A)...")
        baseline = load_baseline_results(f'{dataset_key}_nc')
        for head_label, metrics in baseline.items():
            if metrics:
                metrics['dataset'] = dataset_key
                all_results.append(metrics)
                miou_5c = compute_5c_miou(metrics)
                logger.info(f"    [A-Baseline] {head_label}: 5c-mIoU={miou_5c:.2f}")

        # Groups B, C, D: run experiments
        for group_key in ['B', 'C', 'D']:
            group_info = PROMPT_VARIANTS[group_key]
            config_path = f'./configs/cfg_{dataset_key}_{group_info["suffix"]}.py'

            if not os.path.exists(config_path):
                logger.warning(f"    Config not found: {config_path}")
                continue

            for head_label, use_td, use_ss, _ in HEAD_CONFIGS:
                result = run_single_experiment(
                    dataset_key=dataset_key,
                    head_label=head_label,
                    use_td=use_td,
                    use_ss=use_ss,
                    config_path=config_path,
                    output_dir=output_root,
                    prompt_group=group_info['label'],
                    logger=logger,
                )
                if result:
                    all_results.append(result)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Saving results...")
    csv_path = os.path.join(output_root, 'all_results.csv')
    save_results_csv(csv_path, all_results)
    logger.info(f"All results: {csv_path}")

    # Rank by 5c-mIoU
    logger.info(f"\n{'='*60}")
    logger.info("5-class mIoU Ranking:")
    ranked = []
    for r in all_results:
        miou_5c = compute_5c_miou(r)
        if miou_5c is not None:
            ranked.append((miou_5c, r['dataset'], r['prompt_group'], r.get('config_label', '')))
    ranked.sort(reverse=True)
    for i, (m, ds, grp, cfg) in enumerate(ranked):
        logger.info(f"  {i+1}. [{grp}] {ds}/{cfg}: 5c-mIoU={m:.2f}%")

    # Best prompt per dataset
    logger.info(f"\nBest per dataset:")
    for dataset_key in args.datasets:
        ds_results = [(compute_5c_miou(r), r) for r in all_results if r.get('dataset') == dataset_key]
        ds_results = [(m, r) for m, r in ds_results if m is not None]
        if ds_results:
            ds_results.sort(reverse=True)
            best_m, best_r = ds_results[0]
            logger.info(f"  {dataset_key}: {best_r['prompt_group']}/{best_r.get('config_label','')} = {best_m:.2f}%")

    meta = {
        'phase': '2',
        'description': 'Prompt Engineering — 4 prompt groups (A/B/C/D) × 2 heads',
        'timestamp': timestamp,
        'datasets': args.datasets,
    }
    with open(os.path.join(output_root, 'experiment_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"\nAll output: {output_root}")
    logger.info("Done!")


def parse_args():
    parser = argparse.ArgumentParser(description='Phase 2: Prompt Engineering')
    parser.add_argument('--datasets', nargs='+', default=['vaihingen', 'potsdam'],
                        choices=list(DATASETS.keys()))
    parser.add_argument('--output-root', default='/root/Mynet/autodl-tmp/runs')
    return parser.parse_args()


if __name__ == '__main__':
    main()
