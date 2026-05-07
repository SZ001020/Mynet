_base_ = './cfg_potsdam.py'

# Fast evaluation: 在模型内部预缩放（segearthov3_segmentor.py:160-165）
# Potsdam 6000×6000 → PIL resize to ≤2000 → SAM3 1008² → mask 2000² → upsample 6000²
model = dict(
    slide_stride=0,
    slide_crop=0,
)
