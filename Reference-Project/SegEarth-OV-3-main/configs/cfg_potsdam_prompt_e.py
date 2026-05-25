_base_ = './cfg_potsdam_noclutter.py'
model = dict(
    classname_path='./configs/cls_potsdam_prompt_e.txt',
    bg_idx=0,
)
