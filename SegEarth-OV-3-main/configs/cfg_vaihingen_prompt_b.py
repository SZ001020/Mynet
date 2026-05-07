_base_ = './cfg_vaihingen_noclutter.py'

# Prompt Engineering — B 组
model = dict(
    classname_path='./configs/cls_vaihingen_prompt_b.txt',
    bg_idx=0,
)
