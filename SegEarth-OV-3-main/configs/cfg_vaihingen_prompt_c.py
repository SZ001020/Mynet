_base_ = './cfg_vaihingen_noclutter.py'

# Prompt Engineering — C 组
model = dict(
    classname_path='./configs/cls_vaihingen_prompt_c.txt',
    bg_idx=0,
)
