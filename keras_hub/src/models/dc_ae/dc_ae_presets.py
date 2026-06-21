# Metadata for loading pretrained model weights.
#
# Presets are registered here once the converted weights are uploaded to
# Kaggle/Hugging Face. Until then this stays empty so the model can still be
# constructed and trained from a random initialization. The converter
# `tools/checkpoint_conversion/convert_dc_ae_checkpoints.py` produces the
# `dc_ae_f32c32_in_1.0` preset from `mit-han-lab/dc-ae-f32c32-in-1.0-diffusers`.
backbone_presets = {}
