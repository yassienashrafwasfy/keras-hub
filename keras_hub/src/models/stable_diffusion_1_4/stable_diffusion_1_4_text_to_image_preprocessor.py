import keras
from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.stable_diffusion_1_4.stable_diffusion_1_4_backbone import (  # noqa: E501
    StableDiffusion1_4Backbone,
)
from keras_hub.src.models.text_to_image_preprocessor import (
    TextToImagePreprocessor,
)


@keras_hub_export("keras_hub.models.StableDiffusion1_4TextToImagePreprocessor")
class StableDiffusion1_4TextToImagePreprocessor(TextToImagePreprocessor):
    """Stable Diffusion v1.4 text-to-image model preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.StableDiffusion1_4TextToImage`.

    For use with generation, the layer exposes the `generate_preprocess()`
    method.

    Args:
        clip_preprocessor: A `keras_hub.models.CLIPPreprocessor` instance.
    """

    backbone_cls = StableDiffusion1_4Backbone

    def __init__(self, clip_preprocessor, **kwargs):
        super().__init__(**kwargs)
        self.clip_preprocessor = clip_preprocessor

    @property
    def sequence_length(self):
        """The padded length of model input sequences."""
        return self.clip_preprocessor.sequence_length

    def build(self, input_shape):
        self.built = True

    def generate_preprocess(self, x):
        return self.clip_preprocessor({"prompts": x, "images": None})[
            "token_ids"
        ]

    def get_config(self):
        config = super().get_config()
        config.update(
            {"clip_preprocessor": layers.serialize(self.clip_preprocessor)}
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "clip_preprocessor" in config and isinstance(
            config["clip_preprocessor"], dict
        ):
            config["clip_preprocessor"] = keras.layers.deserialize(
                config["clip_preprocessor"]
            )
        return cls(**config)
