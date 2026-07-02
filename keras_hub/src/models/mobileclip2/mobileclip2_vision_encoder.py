from keras import layers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.mobileclip2.mobileclip2_layers import (
    MobileCLIP2EncoderLayer,
)
from keras_hub.src.models.mobileclip2.mobileclip2_layers import (
    MobileCLIP2VisionEmbedding,
)
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.MobileCLIP2VisionEncoder")
class MobileCLIP2VisionEncoder(Backbone):
    """MobileCLIP2 vision core network with hyperparameters.

    This is a ViT-B/16 style transformer whose patch embedding is a
    convolutional stem (`mci` / `MobileCLIP2ConvStem`) instead of the single
    Conv2D patchify used by standard CLIP. It also follows timm's
    `no_embed_class=True` positional scheme and has no pre-transformer
    LayerNorm.

    Args:
        hidden_dim: int. The size of the transformer hidden state at the end
            of each transformer layer.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        intermediate_activation: activation function. The activation that
            is used for the first Dense layer in a two-layer feedforward network
            for each transformer.
        intermediate_output_index: optional int. The index of the intermediate
            output. If specified, the output will become a dictionary with two
            keys `"sequence_output"` and `"intermediate_output"`.
        stem_channels: tuple of int. The output channels of the conv stem.
        stem_kernel_sizes: tuple of int. The kernel sizes of the conv stem.
        stem_strides: tuple of int. The strides of the conv stem.
        image_shape: tuple. The input shape without the batch size. Defaults to
            `(224, 224, 3)`.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the inputs.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the models computations and weights.
    """

    def __init__(
        self,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        intermediate_activation="gelu",
        intermediate_output_index=None,
        stem_channels=(192, 192, 768),
        stem_kernel_sizes=(4, 2, 2),
        stem_strides=(4, 2, 2),
        image_shape=(224, 224, 3),
        data_format=None,
        dtype=None,
        name=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)
        if data_format == "channels_last":
            height, width = image_shape[0], image_shape[1]
        else:
            height, width = image_shape[1], image_shape[2]
        if height != width:
            raise ValueError(
                "`MobileCLIP2VisionEncoder` expects the height and width to "
                "be the same in `image_shape`. Received: "
                f"image_shape={image_shape}"
            )

        if (
            intermediate_output_index is not None
            and intermediate_output_index < 0
        ):
            intermediate_output_index += num_layers

        # `prefix` is used to prevent duplicate name when utilizing multiple
        # models within a single model.
        prefix = str(name) + "_" if name is not None else ""

        # === Layers ===
        self.embedding = MobileCLIP2VisionEmbedding(
            hidden_dim=hidden_dim,
            image_size=height,
            stem_channels=stem_channels,
            stem_kernel_sizes=stem_kernel_sizes,
            stem_strides=stem_strides,
            data_format=data_format,
            dtype=dtype,
            name=f"{prefix}embedding",
        )
        self.encoder_layers = [
            MobileCLIP2EncoderLayer(
                hidden_dim,
                num_heads,
                intermediate_dim,
                intermediate_activation,
                layer_norm_epsilon=1e-6,
                use_causal_mask=False,  # `False` in the vision encoder.
                dtype=dtype,
                name=f"{prefix}encoder_block_{i}",
            )
            for i in range(num_layers)
        ]
        self.layer_norm = layers.LayerNormalization(
            epsilon=1e-6, dtype=dtype, name=f"{prefix}layer_norm"
        )

        # === Functional Model ===
        image_input = layers.Input(shape=image_shape, name="images")
        x = self.embedding(image_input)
        intermediate_output = None
        for i, block in enumerate(self.encoder_layers):
            x = block(x)
            if i == intermediate_output_index:
                intermediate_output = x
        sequence_output = self.layer_norm(x)

        if intermediate_output_index is not None:
            outputs = {
                "sequence_output": sequence_output,
                "intermediate_output": intermediate_output,
            }
        else:
            outputs = sequence_output
        super().__init__(
            inputs={"images": image_input},
            outputs=outputs,
            dtype=dtype,
            name=name,
            **kwargs,
        )

        # === Config ===
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.intermediate_activation = intermediate_activation
        self.intermediate_output_index = intermediate_output_index
        self.stem_channels = tuple(stem_channels)
        self.stem_kernel_sizes = tuple(stem_kernel_sizes)
        self.stem_strides = tuple(stem_strides)
        self.image_shape = image_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "intermediate_activation": self.intermediate_activation,
                "intermediate_output_index": self.intermediate_output_index,
                "stem_channels": self.stem_channels,
                "stem_kernel_sizes": self.stem_kernel_sizes,
                "stem_strides": self.stem_strides,
                "image_shape": self.image_shape,
            }
        )
        return config
