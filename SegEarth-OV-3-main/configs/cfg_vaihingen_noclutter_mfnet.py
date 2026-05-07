_base_ = './cfg_vaihingen_noclutter.py'

# MFNet standard test split (4 tiles)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ISPRSDataset',
        data_root='/root/autodl-tmp/dataset/Vaihingen',
        ann_file='/root/Mynet/SegEarth-OV-3-main/configs/vaihingen_val_mfnet.txt',
        img_suffix='.tif',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='top', seg_map_path='gts_index'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ]))
