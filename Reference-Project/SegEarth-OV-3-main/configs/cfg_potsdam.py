_base_ = './base_config.py'

# model settings
model = dict(
    classname_path='./configs/cls_potsdam.txt',
    prob_thd=0.1,
    confidence_threshold=0.2,
    bg_idx=5,
    slide_stride=512,
    slide_crop=512,
)

# dataset settings
dataset_type = 'PotsdamDataset'
data_root = '/root/Mynet/autodl-tmp/dataset/Potsdam'

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
        ann_file='/root/Mynet/SegEarth-OV-3-main/configs/potsdam_val.txt',
        img_suffix='_RGB.tif',
        seg_map_suffix='.png',
        data_prefix=dict(
            img_path='2_Ortho_RGB',
            seg_map_path='labels_index'),
        pipeline=test_pipeline))