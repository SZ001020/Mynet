_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_vaihingen.txt',
    prob_thd=0.1,
    bg_idx=5,
    confidence_threshold=0.4,
)

# dataset settings
dataset_type = 'ISPRSDataset'
data_root = '/root/Mynet/autodl-tmp/dataset/Vaihingen'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/root/Mynet/SegEarth-OV-3-main/configs/vaihingen_val.txt',
        img_suffix='.tif',
        seg_map_suffix='.png',
        data_prefix=dict(
            img_path='top',
            seg_map_path='gts_index'),
        pipeline=test_pipeline))