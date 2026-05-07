_base_ = './cfg_vaihingen_noclutter.py'
model = dict(
    classname_path='./configs/cls_vaihingen_prompt_e.txt',
    bg_idx=0,
)
