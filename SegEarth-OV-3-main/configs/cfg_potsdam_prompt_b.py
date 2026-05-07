_base_ = './cfg_potsdam_noclutter.py'

# Prompt Engineering — B 组
model = dict(
    classname_path='./configs/cls_potsdam_prompt_b.txt',
    bg_idx=0,
)
