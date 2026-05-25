_base_ = './cfg_potsdam_noclutter.py'

# MFNet standard test split (6 tiles)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PotsdamDataset',
        data_root='/root/autodl-tmp/dataset/Potsdam',
        ann_file='/root/Mynet/SegEarth-OV-3-main/configs/potsdam_val_mfnet.txt',
        img_suffix='_RGB.tif',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='2_Ortho_RGB', seg_map_path='labels_index'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ]))
