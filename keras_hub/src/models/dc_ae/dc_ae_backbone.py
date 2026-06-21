import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.dc_ae.dc_ae_layers import DCAEDecoder
from keras_hub.src.models.dc_ae.dc_ae_layers import DCAEEncoder
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.DCAEBackbone")
class DCAEBackbone(Backbone):
    """Deep Compression Autoencoder (DC-AE) backbone.

    DC-AE is the deterministic, high-compression autoencoder introduced with
    SANA. Unlike the variational `VAEBackbone` used by Stable Diffusion, DC-AE
    encodes an image directly into a latent (no Gaussian sampling) using
    residual `ResBlock`s, `EfficientViTBlock`s (Sana multi-scale ReLU linear
    attention + gated MBConv), and residual `pixel_unshuffle`/`pixel_shuffle`
    down/upsampling.

    The `dc-ae-f32c32` configuration compresses spatially by `32x` with `32`
    latent channels, so a `512x512x3` image becomes a `16x16x32` latent. This
    enables training latent diffusion models on very small latent grids.

    The default constructor gives a fully customizable, randomly initialized
    model. To load preset architectures and weights, use `from_preset`.

    Args:
        in_channels: int. The number of input image channels. Defaults to `3`.
        latent_channels: int. The number of latent channels. Defaults to `32`.
        attention_head_dim: int. The dimension of each attention head in the
            `EfficientViTBlock`s. Defaults to `32`.
        encoder_block_types: tuple of str. The block type (`"ResBlock"` or
            `"EfficientViTBlock"`) for each encoder stage.
        decoder_block_types: tuple of str. The block type for each decoder
            stage.
        encoder_block_out_channels: tuple of ints. The output channels of each
            encoder stage. The spatial compression ratio is
            `2 ** (len(encoder_block_out_channels) - 1)`.
        decoder_block_out_channels: tuple of ints. The output channels of each
            decoder stage.
        encoder_layers_per_block: tuple of ints. The number of blocks in each
            encoder stage.
        decoder_layers_per_block: tuple of ints. The number of blocks in each
            decoder stage.
        encoder_qkv_multiscales: tuple of tuples. The multi-scale depthwise
            kernel sizes for each encoder `EfficientViTBlock` stage.
        decoder_qkv_multiscales: tuple of tuples. The multi-scale depthwise
            kernel sizes for each decoder `EfficientViTBlock` stage.
        downsample_block_type: str. Either `"pixel_unshuffle"` or `"conv"`.
            Defaults to `"pixel_unshuffle"`.
        upsample_block_type: str. Either `"pixel_shuffle"` or `"interpolate"`.
            Defaults to `"pixel_shuffle"`.
        decoder_norm_types: tuple of str. The norm type (`"batch_norm"` or
            `"rms_norm"`) for each decoder stage.
        decoder_act_fns: tuple of str. The activation for each decoder stage.
        encoder_out_shortcut: bool. Whether the encoder uses the averaged
            channel shortcut into `conv_out`. Defaults to `True`.
        decoder_in_shortcut: bool. Whether the decoder uses the repeated channel
            shortcut into `conv_in`. Defaults to `True`.
        decoder_conv_act_fn: str. The activation before the decoder's final
            convolution. Defaults to `"relu"`.
        scale: float. The latent scaling factor applied when training a
            diffusion model (`latents = encode(images) * scale`). Defaults to
            `0.3189`, the `dc-ae-f32c32-in-1.0` value.
        shift: float. The latent shift factor. Defaults to `0.0`.
        image_shape: tuple. The input image shape without the batch dimension,
            used to build the functional graph. Defaults to `(512, 512, 3)`.
        data_format: `None` or str. Only `"channels_last"` is supported.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the model's computations and weights.

    Example:
    ```python
    # Pretrained DC-AE model.
    model = keras_hub.models.DCAEBackbone.from_preset("dc_ae_f32c32_in_1.0")

    # Encode an image into a 16x16x32 latent and decode it back.
    images = keras.ops.ones((1, 512, 512, 3))
    latents = model.encode(images)
    reconstructed = model.decode(latents)
    ```
    """

    def __init__(
        self,
        in_channels=3,
        latent_channels=32,
        attention_head_dim=32,
        encoder_block_types=(
            "ResBlock",
            "ResBlock",
            "ResBlock",
            "EfficientViTBlock",
            "EfficientViTBlock",
            "EfficientViTBlock",
        ),
        decoder_block_types=(
            "ResBlock",
            "ResBlock",
            "ResBlock",
            "EfficientViTBlock",
            "EfficientViTBlock",
            "EfficientViTBlock",
        ),
        encoder_block_out_channels=(128, 256, 512, 512, 1024, 1024),
        decoder_block_out_channels=(128, 256, 512, 512, 1024, 1024),
        encoder_layers_per_block=(0, 4, 8, 2, 2, 2),
        decoder_layers_per_block=(0, 5, 10, 2, 2, 2),
        encoder_qkv_multiscales=((), (), (), (), (), ()),
        decoder_qkv_multiscales=((), (), (), (), (), ()),
        downsample_block_type="pixel_unshuffle",
        upsample_block_type="pixel_shuffle",
        decoder_norm_types=(
            "batch_norm",
            "batch_norm",
            "batch_norm",
            "rms_norm",
            "rms_norm",
            "rms_norm",
        ),
        decoder_act_fns=("relu", "relu", "relu", "silu", "silu", "silu"),
        encoder_out_shortcut=True,
        decoder_in_shortcut=True,
        decoder_conv_act_fn="relu",
        scale=0.3189,
        shift=0.0,
        image_shape=(512, 512, 3),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)
        if data_format != "channels_last":
            raise NotImplementedError("DC-AE only supports channels_last.")
        compression = 2 ** (len(encoder_block_out_channels) - 1)
        height, width = image_shape[0], image_shape[1]
        if height % compression != 0 or width % compression != 0:
            raise ValueError(
                "height and width in `image_shape` must be divisible by the "
                f"spatial compression ratio ({compression}). Received: "
                f"image_shape={image_shape}"
            )

        # === Layers ===
        self.encoder = DCAEEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            attention_head_dim=attention_head_dim,
            block_types=encoder_block_types,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=encoder_layers_per_block,
            qkv_multiscales=encoder_qkv_multiscales,
            downsample_block_type=downsample_block_type,
            out_shortcut=encoder_out_shortcut,
            data_format=data_format,
            dtype=dtype,
            name="encoder",
        )
        self.decoder = DCAEDecoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            attention_head_dim=attention_head_dim,
            block_types=decoder_block_types,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block,
            qkv_multiscales=decoder_qkv_multiscales,
            norm_types=decoder_norm_types,
            act_fns=decoder_act_fns,
            upsample_block_type=upsample_block_type,
            in_shortcut=decoder_in_shortcut,
            conv_act_fn=decoder_conv_act_fn,
            data_format=data_format,
            dtype=dtype,
            name="decoder",
        )

        # === Functional Model ===
        image_input = keras.Input(shape=image_shape)
        latent = self.encoder(image_input)
        image_output = self.decoder(latent)
        super().__init__(
            inputs=image_input,
            outputs=image_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.attention_head_dim = attention_head_dim
        self.encoder_block_types = encoder_block_types
        self.decoder_block_types = decoder_block_types
        self.encoder_block_out_channels = encoder_block_out_channels
        self.decoder_block_out_channels = decoder_block_out_channels
        self.encoder_layers_per_block = encoder_layers_per_block
        self.decoder_layers_per_block = decoder_layers_per_block
        self.encoder_qkv_multiscales = encoder_qkv_multiscales
        self.decoder_qkv_multiscales = decoder_qkv_multiscales
        self.downsample_block_type = downsample_block_type
        self.upsample_block_type = upsample_block_type
        self.decoder_norm_types = decoder_norm_types
        self.decoder_act_fns = decoder_act_fns
        self.encoder_out_shortcut = encoder_out_shortcut
        self.decoder_in_shortcut = decoder_in_shortcut
        self.decoder_conv_act_fn = decoder_conv_act_fn
        self._scale = scale
        self._shift = shift
        self.image_shape = image_shape
        self.compression_ratio = compression

    @property
    def scale(self):
        """The latent scaling factor used when training a diffusion model."""
        return self._scale

    @property
    def shift(self):
        """The latent shift factor used when training a diffusion model."""
        return self._shift

    def encode(self, inputs, **kwargs):
        """Encode images into the (unscaled) DC-AE latent space."""
        return self.encoder(inputs, **kwargs)

    def decode(self, inputs, **kwargs):
        """Decode DC-AE latents back into images."""
        return self.decoder(inputs, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "latent_channels": self.latent_channels,
                "attention_head_dim": self.attention_head_dim,
                "encoder_block_types": self.encoder_block_types,
                "decoder_block_types": self.decoder_block_types,
                "encoder_block_out_channels": self.encoder_block_out_channels,
                "decoder_block_out_channels": self.decoder_block_out_channels,
                "encoder_layers_per_block": self.encoder_layers_per_block,
                "decoder_layers_per_block": self.decoder_layers_per_block,
                "encoder_qkv_multiscales": self.encoder_qkv_multiscales,
                "decoder_qkv_multiscales": self.decoder_qkv_multiscales,
                "downsample_block_type": self.downsample_block_type,
                "upsample_block_type": self.upsample_block_type,
                "decoder_norm_types": self.decoder_norm_types,
                "decoder_act_fns": self.decoder_act_fns,
                "encoder_out_shortcut": self.encoder_out_shortcut,
                "decoder_in_shortcut": self.decoder_in_shortcut,
                "decoder_conv_act_fn": self.decoder_conv_act_fn,
                "scale": self.scale,
                "shift": self.shift,
                "image_shape": self.image_shape,
            }
        )
        return config
