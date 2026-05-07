#!/usr/bin/env python3
"""
Phase 1 实验运行器 — Zero-shot 基线 + 双头效果分析

对每个数据集运行所有 head × presence 组合，收集完整的 per-class 和 overall 指标。
"""

import os
import sys
import json
import time
import csv
import argparse
from datetime import datetime
from pathlib import Path

# 项目路径
PROJECT_ROOT = '/root/Mynet/SegEarth-OV-3-main'
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.logging import MMLogger

import custom_datasets  # noqa: 注册自定义数据集
import segearthov3_segmentor  # noqa: 注册 SegEarthOV3Segmentation

# 注册返回 per-class 指标的详细 Evaluator
from collections import OrderedDict
from prettytable import PrettyTable
from mmseg.evaluation.metrics import IoUMetric
from mmseg.registry import METRICS
from mmengine.logging import print_log


@METRICS.register_module()
class DetailedIoUMetric(IoUMetric):
    """IoUMetric 扩展：额外返回 per-class IoU 和 Acc"""

    def compute_metrics(self, results: list) -> dict:
        import numpy as np  # noqa: local import for numpy availability
        # results 是 [(intersect, union, pred, label), ...] 列表，需要转置
        results = list(zip(*results))
        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])

        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        class_names = self.dataset_meta['classes']

        # summary
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

        # per-class — 额外返回，用类名做 key
        ret_metrics.pop('aAcc', None)
        for metric_name, metric_arr in ret_metrics.items():
            rounded = np.round(metric_arr * 100, 2)
            for cls_idx, cls_name in enumerate(class_names):
                metrics[f'{metric_name}.{cls_name}'] = rounded[cls_idx]

        # 日志表格
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

DATASETS = {
    'vaihingen': {
        'config': './configs/cfg_vaihingen.py',
        'name': 'ISPRS Vaihingen',
        'classes': ['road', 'building', 'grass', 'tree', 'car', 'clutter'],
    },
    'potsdam': {
        'config': './configs/cfg_potsdam_fast.py',
        'name': 'ISPRS Potsdam',
        'classes': ['road', 'building', 'grass', 'tree', 'car', 'clutter'],
    },
    'loveda': {
        'config': './configs/cfg_loveda.py',
        'name': 'LoveDA',
        'classes': ['background', 'building', 'road', 'water',
                     'barren', 'forest', 'agricultural'],
    },
    'vaihingen_nc': {
        'config': './configs/cfg_vaihingen_noclutter.py',
        'name': 'Vaihingen (no clutter)',
        'classes': ['road', 'building', 'grass', 'tree', 'car'],
    },
    'potsdam_nc': {
        'config': './configs/cfg_potsdam_noclutter.py',
        'name': 'Potsdam (no clutter)',
        'classes': ['road', 'building', 'grass', 'tree', 'car'],
    },
}

# Head 组合配置
HEAD_CONFIGS = [
    # (label, use_transformer_decoder, use_sem_seg, use_presence_score)
    ('Instance-Only',               True,  False, False),
    ('Instance+Presence',           True,  False, True),
    ('Semantic-Only',               False, True,  False),
    ('Semantic+Presence',           False, True,  True),
    ('Dual-Head',                   True,  True,  False),
    ('Dual-Head+Presence (Default)', True,  True,  True),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Phase 1: Zero-shot baseline experiments')
    parser.add_argument('--datasets', nargs='+', default=['vaihingen', 'potsdam', 'loveda'],
                        choices=list(DATASETS.keys()), help='Datasets to evaluate')
    parser.add_argument('--configs', nargs='+', default=None,
                        help='Specific head configs to run (0-5), e.g. 0 1 5. Default: all')
    parser.add_argument('--output-root', default='/root/Mynet/autodl-tmp/runs',
                        help='Root output directory')
    return parser.parse_args()


def build_cfg(base_cfg_path, use_transformer_decoder, use_sem_seg, use_presence_score, work_dir):
    """构建实验配置"""
    cfg = Config.fromfile(base_cfg_path)

    # 覆盖模型参数
    cfg.model.use_transformer_decoder = use_transformer_decoder
    cfg.model.use_sem_seg = use_sem_seg
    cfg.model.use_presence_score = use_presence_score

    # 使用返回 per-class 指标的 evaluator
    cfg.test_evaluator = dict(
        type='DetailedIoUMetric',
        iou_metrics=['mIoU'],
    )

    cfg.work_dir = work_dir
    cfg.log_level = 'INFO'

    return cfg


def _to_native(val):
    """递归转换 numpy 类型为原生 Python 类型"""
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
    """从 mmengine test 结果中提取完整的指标"""
    import numpy as np
    metrics = {}

    # Overall metrics (标准 mmseg IoUMetric 输出)
    metrics['aAcc'] = _to_native(results.get('aAcc', 0))
    metrics['mIoU'] = _to_native(results.get('mIoU', 0))
    metrics['mAcc'] = _to_native(results.get('mAcc', 0))

    # 收集所有数字类型的 per-class 指标
    known_meta_keys = {'aAcc', 'mIoU', 'mAcc', 'IoU', 'Acc'}
    for key, val in results.items():
        if not isinstance(key, str):
            continue
        if key in known_meta_keys:
            continue
        if isinstance(val, (int, float, np.integer, np.floating, np.ndarray)):
            v = _to_native(val)
            # 接受 0-100 范围的 per-class 百分比值
            if isinstance(v, (int, float)) and -0.01 < v < 101.0:
                metrics[key] = v

    return metrics


def save_results_csv(csv_path, all_rows):
    """保存实验结果到 CSV"""
    if not all_rows:
        return

    # 收集所有可能的列名
    all_keys = set()
    for row in all_rows:
        all_keys.update(row.keys())

    # 固定顺序：meta 列在前，metric 列在后
    meta_keys = ['dataset', 'config_label', 'use_transformer_decoder',
                 'use_sem_seg', 'use_presence_score', 'timestamp']
    metric_keys = sorted([k for k in all_keys if k not in meta_keys])

    # 把 aAcc, mIoU, mAcc 放前面
    priority = ['aAcc', 'mIoU', 'mAcc']
    metric_keys = [k for k in priority if k in metric_keys] + \
                  [k for k in metric_keys if k not in priority]

    fieldnames = meta_keys + metric_keys

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            # 确保所有键都存在
            for k in fieldnames:
                row.setdefault(k, '')
            writer.writerow({k: row[k] for k in fieldnames})


def run_single_experiment(dataset_key, head_label, use_td, use_ss, use_ps,
                          base_config_path, output_dir, logger):
    """运行单个实验配置"""
    import gc
    import torch

    exp_name = f"{dataset_key}_{head_label.replace(' ', '_').replace('+', '-').replace('(', '').replace(')', '')}"
    work_dir = os.path.join(output_dir, exp_name)
    os.makedirs(work_dir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"Experiment: {dataset_key} | {head_label}")
    logger.info(f"  Transformer Decoder: {use_td}")
    logger.info(f"  Semantic Seg: {use_ss}")
    logger.info(f"  Presence Score: {use_ps}")
    logger.info(f"  Work dir: {work_dir}")

    start_time = time.time()
    runner = None
    try:
        cfg = build_cfg(base_config_path, use_td, use_ss, use_ps, work_dir)
        runner = Runner.from_cfg(cfg)
        results = runner.test()

        elapsed = time.time() - start_time
        metrics = extract_metrics(results)

        # 添加 meta 信息
        metrics['dataset'] = DATASETS[dataset_key]['name']
        metrics['config_label'] = head_label
        metrics['use_transformer_decoder'] = use_td
        metrics['use_sem_seg'] = use_ss
        metrics['use_presence_score'] = use_ps
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['elapsed_seconds'] = round(elapsed, 1)

        logger.info(f"  Results: aAcc={metrics['aAcc']:.3f}, "
                     f"mIoU={metrics['mIoU']:.3f}, mAcc={metrics['mAcc']:.3f}")
        logger.info(f"  Time: {elapsed:.1f}s")

        # 单独保存此实验的详细结果
        detail_path = os.path.join(work_dir, 'metrics.json')
        with open(detail_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"  FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return {
            'dataset': DATASETS[dataset_key]['name'],
            'config_label': head_label,
            'use_transformer_decoder': use_td,
            'use_sem_seg': use_ss,
            'use_presence_score': use_ps,
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': round(elapsed, 1),
            'error': str(e),
        }
    finally:
        # 释放显存
        if runner is not None:
            del runner
        gc.collect()
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output_root, f'phase1_baseline_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    # 设置日志
    log_path = os.path.join(output_root, 'experiment.log')
    logger = MMLogger.get_instance('phase1', log_file=log_path,
                                    file_mode='w', log_level='INFO')
    logger.info(f"Phase 1: Zero-shot Baseline + Dual-Head Analysis")
    logger.info(f"Output: {output_root}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Time: {datetime.now().isoformat()}")

    # 确定要运行的配置
    if args.configs is not None:
        config_indices = [int(c) for c in args.configs]
        head_configs = [HEAD_CONFIGS[i] for i in config_indices]
    else:
        head_configs = HEAD_CONFIGS

    logger.info(f"Configurations to run: {[h[0] for h in head_configs]}")

    all_results = []

    for dataset_key in args.datasets:
        ds_info = DATASETS[dataset_key]
        logger.info(f"\n{'#'*60}")
        logger.info(f"Dataset: {ds_info['name']} ({dataset_key})")
        logger.info(f"Classes: {ds_info['classes']}")

        for head_label, use_td, use_ss, use_ps in head_configs:
            result = run_single_experiment(
                dataset_key=dataset_key,
                head_label=head_label,
                use_td=use_td,
                use_ss=use_ss,
                use_ps=use_ps,
                base_config_path=ds_info['config'],
                output_dir=output_root,
                logger=logger,
            )
            all_results.append(result)

    # ============================================================
    # 保存汇总结果
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info("All experiments complete. Saving summary...")

    # 完整 CSV
    csv_path = os.path.join(output_root, 'all_results.csv')
    save_results_csv(csv_path, all_results)
    logger.info(f"Results saved to: {csv_path}")

    # 按数据集分组摘要
    for dataset_key in args.datasets:
        ds_name = DATASETS[dataset_key]['name']
        ds_results = [r for r in all_results if r.get('dataset') == ds_name]
        ds_csv_path = os.path.join(output_root, f'{dataset_key}_results.csv')
        save_results_csv(ds_csv_path, ds_results)
        logger.info(f"{dataset_key} results saved to: {ds_csv_path}")

        # 打印最佳配置
        valid = [r for r in ds_results if 'mIoU' in r]
        if valid:
            best = max(valid, key=lambda r: r['mIoU'])
            logger.info(f"  {dataset_key} best: {best['config_label']} "
                         f"mIoU={best['mIoU']:.3f}, aAcc={best['aAcc']:.3f}")

    # 保存实验元信息
    meta = {
        'phase': '1',
        'description': 'Zero-shot baseline + dual-head analysis',
        'timestamp': timestamp,
        'datasets': args.datasets,
        'head_configs': [{'label': h[0], 'td': h[1], 'sem': h[2], 'pres': h[3]}
                          for h in head_configs],
        'total_experiments': len(all_results),
    }
    with open(os.path.join(output_root, 'experiment_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"\nAll results in: {output_root}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
