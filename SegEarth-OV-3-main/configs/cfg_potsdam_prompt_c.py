_base_ = './cfg_potsdam_noclutter.py'

# Prompt Engineering — C 组
model = dict(
    classname_path='./configs/cls_potsdam_prompt_c.txt',
    bg_idx=0,
)
