import math

import keras
from keras import layers
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.utils.keras_utils import fused_attention_op_available
from keras_hub.src.utils.keras_utils import standardize_data_format


class Timesteps(layers.Layer):
    """Sinusoidal timestep embedding matching `diffusers`.

    Uses `flip_sin_to_cos=True` and `downscale_freq_shift=0`, the settings used
    by the Stable Diffusion v1.4 UNet.
    """

    def __init__(self, embedding_dim, max_period=10000, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = int(embedding_dim)
        self.max_period = float(max_period)
        half_dim = self.embedding_dim // 2
        exponent = ops.multiply(
            -math.log(max_period),
            ops.arange(0, half_dim, dtype="float32"),
        )
        self.freq = ops.exp(ops.divide(exponent, half_dim))

    def call(self, inputs):
        compute_dtype = keras.backend.result_type(self.compute_dtype, "float32")
        x = ops.cast(inputs, compute_dtype)
        freqs = ops.cast(self.freq, compute_dtype)
        x = ops.multiply(ops.expand_dims(x, axis=-1), ops.expand_dims(freqs, 0))
        # `flip_sin_to_cos=True` places the cosine terms first.
        embedding = ops.concatenate([ops.cos(x), ops.sin(x)], axis=-1)
        return ops.cast(embedding, self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "max_period": self.max_period,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        return tuple(inputs_shape) + (self.embedding_dim,)


class TimestepEmbedding(layers.Layer):
    """Two-layer MLP that projects the sinusoidal embedding."""

    def __init__(self, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = int(embedding_dim)
        self.linear_1 = layers.Dense(
            embedding_dim, dtype=self.dtype_policy, name="linear_1"
        )
        self.act = layers.Activation("silu", dtype=self.dtype_policy)
        self.linear_2 = layers.Dense(
            embedding_dim, dtype=self.dtype_policy, name="linear_2"
        )

    def build(self, inputs_shape):
        self.linear_1.build(inputs_shape)
        inputs_shape = self.linear_1.compute_output_shape(inputs_shape)
        self.linear_2.build(inputs_shape)

    def call(self, inputs, training=None):
        x = self.linear_1(inputs, training=training)
        x = self.act(x)
        return self.linear_2(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"embedding_dim": self.embedding_dim})
        return config

    def compute_output_shape(self, inputs_shape):
        return tuple(inputs_shape[:-1]) + (self.embedding_dim,)


class CrossAttention(layers.Layer):
    """Multi-head attention with separate query / context projections.

    The projection layout (`to_q`/`to_k`/`to_v` without bias, `to_out` with
    bias) matches `diffusers` so weights port directly.
    """

    def __init__(self, num_heads, head_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self._inverse_sqrt_head_dim = 1.0 / math.sqrt(float(head_dim))

        self.to_q = layers.Dense(
            self.inner_dim, use_bias=False, dtype=self.dtype_policy, name="to_q"
        )
        self.to_k = layers.Dense(
            self.inner_dim, use_bias=False, dtype=self.dtype_policy, name="to_k"
        )
        self.to_v = layers.Dense(
            self.inner_dim, use_bias=False, dtype=self.dtype_policy, name="to_v"
        )
        self.to_out = layers.Dense(
            self.inner_dim, dtype=self.dtype_policy, name="to_out"
        )
        self.softmax = layers.Softmax(dtype="float32")

    def build(self, inputs_shape, context_shape=None):
        context_shape = context_shape or inputs_shape
        self.to_q.build(inputs_shape)
        self.to_k.build(context_shape)
        self.to_v.build(context_shape)
        self.to_out.build((None, None, self.inner_dim))

    def _compute_attention(self, query, key, value):
        batch_size = ops.shape(query)[0]
        if fused_attention_op_available():
            encoded = ops.dot_product_attention(
                query, key, value, scale=self._inverse_sqrt_head_dim
            )
            return ops.reshape(encoded, (batch_size, -1, self.inner_dim))
        logits = ops.einsum("BTNH,BSNH->BNTS", query, key)
        logits = ops.multiply(logits, self._inverse_sqrt_head_dim)
        probs = ops.cast(self.softmax(logits), self.compute_dtype)
        encoded = ops.einsum("BNTS,BSNH->BTNH", probs, value)
        return ops.reshape(encoded, (batch_size, -1, self.inner_dim))

    def call(self, inputs, context=None, training=None):
        context = inputs if context is None else context
        query = self.to_q(inputs, training=training)
        key = self.to_k(context, training=training)
        value = self.to_v(context, training=training)

        batch_size = ops.shape(query)[0]
        query = ops.reshape(
            query, (batch_size, -1, self.num_heads, self.head_dim)
        )
        key = ops.reshape(key, (batch_size, -1, self.num_heads, self.head_dim))
        value = ops.reshape(
            value, (batch_size, -1, self.num_heads, self.head_dim)
        )
        attention = self._compute_attention(query, key, value)
        return self.to_out(attention, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "head_dim": self.head_dim})
        return config

    def compute_output_shape(self, inputs_shape, context_shape=None):
        return tuple(inputs_shape[:-1]) + (self.inner_dim,)


class GEGLUFeedForward(layers.Layer):
    """A GEGLU feed-forward network used in the transformer blocks."""

    def __init__(self, expansion_factor=4, **kwargs):
        super().__init__(**kwargs)
        self.expansion_factor = int(expansion_factor)

    def build(self, inputs_shape):
        dim = inputs_shape[-1]
        inner_dim = dim * self.expansion_factor
        self.proj = layers.Dense(
            inner_dim * 2, dtype=self.dtype_policy, name="proj"
        )
        self.proj.build(inputs_shape)
        self.output_dense = layers.Dense(
            dim, dtype=self.dtype_policy, name="output_dense"
        )
        self.output_dense.build((None, None, inner_dim))

    def call(self, inputs, training=None):
        x = self.proj(inputs, training=training)
        x, gate = ops.split(x, 2, axis=-1)
        x = ops.multiply(x, ops.gelu(gate, approximate=False))
        return self.output_dense(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"expansion_factor": self.expansion_factor})
        return config

    def compute_output_shape(self, inputs_shape):
        return inputs_shape


class BasicTransformerBlock(layers.Layer):
    """Self-attention, cross-attention and GEGLU feed-forward block."""

    def __init__(self, num_heads, head_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)

        self.norm1 = layers.LayerNormalization(
            epsilon=1e-5, dtype="float32", name="norm1"
        )
        self.attn1 = CrossAttention(
            num_heads, head_dim, dtype=self.dtype_policy, name="attn1"
        )
        self.norm2 = layers.LayerNormalization(
            epsilon=1e-5, dtype="float32", name="norm2"
        )
        self.attn2 = CrossAttention(
            num_heads, head_dim, dtype=self.dtype_policy, name="attn2"
        )
        self.norm3 = layers.LayerNormalization(
            epsilon=1e-5, dtype="float32", name="norm3"
        )
        self.ff = GEGLUFeedForward(dtype=self.dtype_policy, name="ff")

    def build(self, inputs_shape, context_shape):
        self.norm1.build(inputs_shape)
        self.attn1.build(inputs_shape)
        self.norm2.build(inputs_shape)
        self.attn2.build(inputs_shape, context_shape)
        self.norm3.build(inputs_shape)
        self.ff.build(inputs_shape)

    def call(self, inputs, context, training=None):
        x = inputs
        x = ops.add(self.attn1(self.norm1(x), training=training), x)
        x = ops.add(
            self.attn2(self.norm2(x), context=context, training=training), x
        )
        x = ops.add(self.ff(self.norm3(x), training=training), x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "head_dim": self.head_dim})
        return config

    def compute_output_shape(self, inputs_shape, context_shape):
        return inputs_shape


class Transformer2DModel(layers.Layer):
    """A spatial transformer that conditions features on the text context."""

    def __init__(self, num_heads, head_dim, data_format=None, **kwargs):
        super().__init__(**kwargs)
        data_format = standardize_data_format(data_format)
        if data_format != "channels_last":
            raise NotImplementedError
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim

        self.norm = layers.GroupNormalization(
            groups=32,
            axis=-1,
            epsilon=1e-6,
            dtype=self.dtype_policy,
            name="norm",
        )
        self.proj_in = layers.Conv2D(
            self.inner_dim, 1, dtype=self.dtype_policy, name="proj_in"
        )
        self.transformer_block = BasicTransformerBlock(
            num_heads,
            head_dim,
            dtype=self.dtype_policy,
            name="transformer_block",
        )
        self.proj_out = layers.Conv2D(
            self.inner_dim, 1, dtype=self.dtype_policy, name="proj_out"
        )

    def build(self, inputs_shape, context_shape):
        self.norm.build(inputs_shape)
        self.proj_in.build(inputs_shape)
        inputs_shape = self.proj_in.compute_output_shape(inputs_shape)
        sequence_shape = (inputs_shape[0], None, self.inner_dim)
        self.transformer_block.build(sequence_shape, context_shape)
        self.proj_out.build(inputs_shape)

    def call(self, inputs, context, training=None):
        residual = inputs
        x = self.norm(inputs, training=training)
        x = self.proj_in(x, training=training)
        batch_size = ops.shape(x)[0]
        height = ops.shape(x)[1]
        width = ops.shape(x)[2]
        x = ops.reshape(x, (batch_size, height * width, self.inner_dim))
        x = self.transformer_block(x, context, training=training)
        x = ops.reshape(x, (batch_size, height, width, self.inner_dim))
        x = self.proj_out(x, training=training)
        return ops.add(x, residual)

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "head_dim": self.head_dim})
        return config

    def compute_output_shape(self, inputs_shape, context_shape):
        return inputs_shape


class ResnetBlock2D(layers.Layer):
    """A ResNet block that injects the timestep embedding."""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)

        self.norm1 = layers.GroupNormalization(
            groups=32,
            axis=-1,
            epsilon=1e-5,
            dtype=self.dtype_policy,
            name="norm1",
        )
        self.conv1 = layers.Conv2D(
            self.filters,
            3,
            padding="same",
            dtype=self.dtype_policy,
            name="conv1",
        )
        self.time_emb_proj = layers.Dense(
            self.filters, dtype=self.dtype_policy, name="time_emb_proj"
        )
        self.norm2 = layers.GroupNormalization(
            groups=32,
            axis=-1,
            epsilon=1e-5,
            dtype=self.dtype_policy,
            name="norm2",
        )
        self.conv2 = layers.Conv2D(
            self.filters,
            3,
            padding="same",
            dtype=self.dtype_policy,
            name="conv2",
        )
        self.act = layers.Activation("silu", dtype=self.dtype_policy)

    def build(self, inputs_shape, time_emb_shape):
        in_channels = inputs_shape[-1]
        self.norm1.build(inputs_shape)
        self.conv1.build(inputs_shape)
        self.time_emb_proj.build(time_emb_shape)
        hidden_shape = self.conv1.compute_output_shape(inputs_shape)
        self.norm2.build(hidden_shape)
        self.conv2.build(hidden_shape)
        self.conv_shortcut = None
        if in_channels != self.filters:
            self.conv_shortcut = layers.Conv2D(
                self.filters, 1, dtype=self.dtype_policy, name="conv_shortcut"
            )
            self.conv_shortcut.build(inputs_shape)

    def call(self, inputs, time_emb, training=None):
        x = self.norm1(inputs, training=training)
        x = self.act(x)
        x = self.conv1(x, training=training)
        time_emb = self.time_emb_proj(self.act(time_emb), training=training)
        x = ops.add(x, time_emb[:, None, None, :])
        x = self.norm2(x, training=training)
        x = self.act(x)
        x = self.conv2(x, training=training)
        residual = inputs
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(inputs, training=training)
        return ops.add(x, residual)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    def compute_output_shape(self, inputs_shape, time_emb_shape):
        return tuple(inputs_shape[:-1]) + (self.filters,)


class Downsample2D(layers.Layer):
    """Strided convolution downsampler with symmetric padding."""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.pad = layers.ZeroPadding2D(padding=1, dtype=self.dtype_policy)
        self.conv = layers.Conv2D(
            self.filters,
            3,
            strides=2,
            padding="valid",
            dtype=self.dtype_policy,
            name="conv",
        )

    def build(self, inputs_shape):
        inputs_shape = self.pad.compute_output_shape(inputs_shape)
        self.conv.build(inputs_shape)

    def call(self, inputs, training=None):
        return self.conv(self.pad(inputs), training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    def compute_output_shape(self, inputs_shape):
        inputs_shape = self.pad.compute_output_shape(inputs_shape)
        return self.conv.compute_output_shape(inputs_shape)


class Upsample2D(layers.Layer):
    """Nearest-neighbor upsampler followed by a convolution."""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.upsample = layers.UpSampling2D(
            2, interpolation="nearest", dtype=self.dtype_policy
        )
        self.conv = layers.Conv2D(
            self.filters,
            3,
            padding="same",
            dtype=self.dtype_policy,
            name="conv",
        )

    def build(self, inputs_shape):
        inputs_shape = self.upsample.compute_output_shape(inputs_shape)
        self.conv.build(inputs_shape)

    def call(self, inputs, training=None):
        return self.conv(self.upsample(inputs), training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    def compute_output_shape(self, inputs_shape):
        inputs_shape = self.upsample.compute_output_shape(inputs_shape)
        return self.conv.compute_output_shape(inputs_shape)


class DownBlock2D(layers.Layer):
    """A down block of ResNet layers, optionally with cross-attention."""

    def __init__(
        self,
        filters,
        num_layers,
        num_heads,
        use_attention,
        add_downsample,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.use_attention = bool(use_attention)
        self.add_downsample = bool(add_downsample)

        self.resnets = [
            ResnetBlock2D(filters, dtype=self.dtype_policy, name=f"resnet_{i}")
            for i in range(num_layers)
        ]
        self.attentions = []
        if use_attention:
            self.attentions = [
                Transformer2DModel(
                    num_heads,
                    filters // num_heads,
                    dtype=self.dtype_policy,
                    name=f"attention_{i}",
                )
                for i in range(num_layers)
            ]
        self.downsample = None
        if add_downsample:
            self.downsample = Downsample2D(
                filters, dtype=self.dtype_policy, name="downsample"
            )

    def build(self, inputs_shape, time_emb_shape, context_shape=None):
        shape = inputs_shape
        for i in range(self.num_layers):
            self.resnets[i].build(shape, time_emb_shape)
            shape = self.resnets[i].compute_output_shape(shape, time_emb_shape)
            if self.use_attention:
                self.attentions[i].build(shape, context_shape)
        if self.downsample is not None:
            self.downsample.build(shape)

    def call(self, inputs, time_emb, context=None, training=None):
        x = inputs
        states = []
        for i in range(self.num_layers):
            x = self.resnets[i](x, time_emb, training=training)
            if self.use_attention:
                x = self.attentions[i](x, context, training=training)
            states.append(x)
        if self.downsample is not None:
            x = self.downsample(x, training=training)
            states.append(x)
        return x, states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "use_attention": self.use_attention,
                "add_downsample": self.add_downsample,
            }
        )
        return config


class MidBlock2D(layers.Layer):
    """The middle block: ResNet, cross-attention, ResNet."""

    def __init__(self, filters, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.num_heads = int(num_heads)
        self.resnet_0 = ResnetBlock2D(
            filters, dtype=self.dtype_policy, name="resnet_0"
        )
        self.attention = Transformer2DModel(
            num_heads,
            filters // num_heads,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.resnet_1 = ResnetBlock2D(
            filters, dtype=self.dtype_policy, name="resnet_1"
        )

    def build(self, inputs_shape, time_emb_shape, context_shape):
        self.resnet_0.build(inputs_shape, time_emb_shape)
        shape = self.resnet_0.compute_output_shape(inputs_shape, time_emb_shape)
        self.attention.build(shape, context_shape)
        self.resnet_1.build(shape, time_emb_shape)

    def call(self, inputs, time_emb, context, training=None):
        x = self.resnet_0(inputs, time_emb, training=training)
        x = self.attention(x, context, training=training)
        x = self.resnet_1(x, time_emb, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "num_heads": self.num_heads})
        return config

    def compute_output_shape(self, inputs_shape, time_emb_shape, context_shape):
        return tuple(inputs_shape[:-1]) + (self.filters,)


class UpBlock2D(layers.Layer):
    """An up block of ResNet layers, optionally with cross-attention.

    Each ResNet consumes a skip connection from the matching down block.
    """

    def __init__(
        self,
        filters,
        num_layers,
        num_heads,
        use_attention,
        add_upsample,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.use_attention = bool(use_attention)
        self.add_upsample = bool(add_upsample)

        self.resnets = [
            ResnetBlock2D(filters, dtype=self.dtype_policy, name=f"resnet_{i}")
            for i in range(num_layers)
        ]
        self.attentions = []
        if use_attention:
            self.attentions = [
                Transformer2DModel(
                    num_heads,
                    filters // num_heads,
                    dtype=self.dtype_policy,
                    name=f"attention_{i}",
                )
                for i in range(num_layers)
            ]
        self.upsample = None
        if add_upsample:
            self.upsample = Upsample2D(
                filters, dtype=self.dtype_policy, name="upsample"
            )

    def build(
        self,
        inputs_shape,
        time_emb_shape,
        res_samples_shape,
        context_shape=None,
    ):
        for i in range(self.num_layers):
            channels = inputs_shape[-1] + res_samples_shape[i][-1]
            shape = tuple(inputs_shape[:-1]) + (channels,)
            self.resnets[i].build(shape, time_emb_shape)
            inputs_shape = self.resnets[i].compute_output_shape(
                shape, time_emb_shape
            )
            if self.use_attention:
                self.attentions[i].build(inputs_shape, context_shape)
        if self.upsample is not None:
            self.upsample.build(inputs_shape)

    def call(self, inputs, time_emb, res_samples, context=None, training=None):
        x = inputs
        for i in range(self.num_layers):
            x = ops.concatenate([x, res_samples[i]], axis=-1)
            x = self.resnets[i](x, time_emb, training=training)
            if self.use_attention:
                x = self.attentions[i](x, context, training=training)
        if self.upsample is not None:
            x = self.upsample(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "use_attention": self.use_attention,
                "add_upsample": self.add_upsample,
            }
        )
        return config


class UNet(Backbone):
    """The UNet denoising network for Stable Diffusion v1.4.

    Predicts the noise residual added to a latent given the diffusion timestep
    and the encoded text context.

    Args:
        in_channels: int. The number of channels in the input latent.
        out_channels: int. The number of channels in the predicted noise.
        block_out_channels: list of ints. The number of filters in each
            down/up block.
        layers_per_block: int. The number of ResNet layers per block.
        num_heads: int. The number of attention heads in each transformer.
        cross_attention_dim: int. The dimension of the text context.
        context_max_length: int. The sequence length of the text context.
        data_format: `None` or str. Only `"channels_last"` is supported.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the model's computations and weights.
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=2,
        num_heads=8,
        cross_attention_dim=768,
        context_max_length=77,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)
        if data_format != "channels_last":
            raise NotImplementedError
        block_out_channels = tuple(block_out_channels)
        time_embed_dim = block_out_channels[0] * 4
        num_blocks = len(block_out_channels)

        # === Layers ===
        self.time_proj = Timesteps(
            block_out_channels[0], dtype=dtype, name="time_proj"
        )
        self.time_embedding = TimestepEmbedding(
            time_embed_dim, dtype=dtype, name="time_embedding"
        )
        self.conv_in = layers.Conv2D(
            block_out_channels[0],
            3,
            padding="same",
            dtype=dtype,
            name="conv_in",
        )

        # Down blocks: cross-attention for all but the last block.
        self.down_blocks = []
        for i in range(num_blocks):
            self.down_blocks.append(
                DownBlock2D(
                    block_out_channels[i],
                    layers_per_block,
                    num_heads,
                    use_attention=(i != num_blocks - 1),
                    add_downsample=(i != num_blocks - 1),
                    dtype=dtype,
                    name=f"down_block_{i}",
                )
            )

        self.mid_block = MidBlock2D(
            block_out_channels[-1], num_heads, dtype=dtype, name="mid_block"
        )

        # Up blocks mirror the down blocks: the first up block has no attention.
        reversed_channels = list(reversed(block_out_channels))
        self.up_blocks = []
        for i in range(num_blocks):
            self.up_blocks.append(
                UpBlock2D(
                    reversed_channels[i],
                    layers_per_block + 1,
                    num_heads,
                    use_attention=(i != 0),
                    add_upsample=(i != num_blocks - 1),
                    dtype=dtype,
                    name=f"up_block_{i}",
                )
            )

        self.conv_norm_out = layers.GroupNormalization(
            groups=32, axis=-1, epsilon=1e-5, dtype=dtype, name="conv_norm_out"
        )
        self.conv_act = layers.Activation("silu", dtype=dtype)
        self.conv_out = layers.Conv2D(
            out_channels, 3, padding="same", dtype=dtype, name="conv_out"
        )

        # === Functional Model ===
        latent_input = keras.Input(
            shape=(None, None, in_channels), name="latent"
        )
        timestep_input = keras.Input(shape=(), name="timestep")
        context_input = keras.Input(
            shape=(None, cross_attention_dim), name="context"
        )

        time_emb = self.time_proj(timestep_input)
        time_emb = self.time_embedding(time_emb)

        x = self.conv_in(latent_input)
        res_samples = [x]
        for down_block in self.down_blocks:
            x, states = down_block(x, time_emb, context_input)
            res_samples.extend(states)

        x = self.mid_block(x, time_emb, context_input)

        for up_block in self.up_blocks:
            num = up_block.num_layers
            # Consume the most recent skips first, matching the down path.
            skips = list(reversed(res_samples[-num:]))
            res_samples = res_samples[:-num]
            x = up_block(x, time_emb, skips, context_input)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        outputs = self.conv_out(x)

        super().__init__(
            inputs={
                "latent": latent_input,
                "timestep": timestep_input,
                "context": context_input,
            },
            outputs=outputs,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.num_heads = num_heads
        self.cross_attention_dim = cross_attention_dim
        self.context_max_length = context_max_length

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "block_out_channels": self.block_out_channels,
                "layers_per_block": self.layers_per_block,
                "num_heads": self.num_heads,
                "cross_attention_dim": self.cross_attention_dim,
                "context_max_length": self.context_max_length,
            }
        )
        return config
