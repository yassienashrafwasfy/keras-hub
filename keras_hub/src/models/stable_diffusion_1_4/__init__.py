from keras_hub.src.models.stable_diffusion_1_4.stable_diffusion_1_4_backbone import (  # noqa: E501
    StableDiffusion1_4Backbone,
)
from keras_hub.src.models.stable_diffusion_1_4.stable_diffusion_1_4_presets import (  # noqa: E501
    backbone_presets,
)
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, StableDiffusion1_4Backbone)
