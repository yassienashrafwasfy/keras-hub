from keras_hub.src.models.mobileclip2.mobileclip2_backbone import (
    MobileCLIP2Backbone,
)
from keras_hub.src.models.mobileclip2.mobileclip2_presets import (
    backbone_presets,
)
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, MobileCLIP2Backbone)
