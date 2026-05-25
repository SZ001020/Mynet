_base_ = './cfg_potsdam_noclutter.py'

# Prompt Engineering — D 组
model = dict(
    classname_path='./configs/cls_potsdam_prompt_d.txt',
    bg_idx=0,
)
