_base_ = './cfg_potsdam_fast.py'

# 去除 clutter 类别，仅评估 5 个主要类别
model = dict(
    classname_path='./configs/cls_potsdam_noclutter.txt',
    bg_idx=0,
)
