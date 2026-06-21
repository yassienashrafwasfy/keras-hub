from keras_hub.src.models.dc_ae.dc_ae_backbone import DCAEBackbone
from keras_hub.src.models.dc_ae.dc_ae_presets import backbone_presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, DCAEBackbone)
