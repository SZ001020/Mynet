_base_ = './cfg_vaihingen_noclutter.py'

# Prompt Engineering — D 组
model = dict(
    classname_path='./configs/cls_vaihingen_prompt_d.txt',
    bg_idx=0,
)
