from keras_hub.src.models.stable_diffusion_2_1.stable_diffusion_2_1_backbone import (  # noqa: E501
    StableDiffusion2_1Backbone,
)
from keras_hub.src.models.stable_diffusion_2_1.stable_diffusion_2_1_presets import (  # noqa: E501
    backbone_presets,
)
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, StableDiffusion2_1Backbone)
