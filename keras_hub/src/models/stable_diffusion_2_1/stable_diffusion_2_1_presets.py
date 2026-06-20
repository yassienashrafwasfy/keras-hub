# Presets are registered here once the converted weights are uploaded to
# Kaggle. Until then this stays empty (matching `stable_diffusion_1_4`) so that
# `from_preset` never points at an unpublished handle. The two intended presets
# are `stable_diffusion_2_1` (768x768, v_prediction) and
# `stable_diffusion_2_1_base` (512x512, epsilon); their architecture is defined
# by `tools/checkpoint_conversion/convert_stable_diffusion_2_1_checkpoints.py`.
backbone_presets = {}
