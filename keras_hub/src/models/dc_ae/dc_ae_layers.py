import keras
from keras import ops

from keras_hub.src.utils.keras_utils import standardize_data_format


def pixel_unshuffle(x, factor):
    """Channels-last equivalent of `torch.nn.functional.pixel_unshuffle`.

    `(B, H, W, C) -> (B, H // factor, W // factor, C * factor ** 2)`. The output
    channel ordering matches PyTorch: channel `c * factor ** 2 + i * factor + j`
    holds the `(i, j)` offset within each `factor x factor` spatial block.
    """
    shape = ops.shape(x)
    b, height, width = shape[0], shape[1], shape[2]
    channels = x.shape[-1]
    new_height = height // factor
    new_width = width // factor
    x = ops.reshape(x, (b, new_height, factor, new_width, factor, channels))
    # (b, new_height, new_width, channels, factor, factor)
    x = ops.transpose(x, (0, 1, 3, 5, 2, 4))
    x = ops.reshape(x, (b, new_height, new_width, channels * factor * factor))
    return x


def pixel_shuffle(x, factor):
    """Channels-last equivalent of `torch.nn.functional.pixel_shuffle`.

    `(B, H, W, C * factor ** 2) -> (B, H * factor, W * factor, C)`, inverse of
    `pixel_unshuffle`.
    """
    shape = ops.shape(x)
    b, height, width = shape[0], shape[1], shape[2]
    channels = x.shape[-1]
    new_channels = channels // (factor * factor)
    x = ops.reshape(x, (b, height, width, new_channels, factor, factor))
    # (b, height, factor, width, factor, new_channels)
    x = ops.transpose(x, (0, 1, 4, 2, 5, 3))
    x = ops.reshape(x, (b, height * factor, width * factor, new_channels))
    return x


class RMSNorm2D(keras.layers.Layer):
    """Root-mean-square normalization over the channel (last) axis.

    Matches diffusers' `RMSNorm(dim, eps, elementwise_affine=True, bias=True)`
    applied to the channel dimension of a `channels_last` feature map.
    """

    def __init__(self, channels, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.channels,),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(self.channels,),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        x = ops.cast(inputs, "float32")
        variance = ops.mean(ops.square(x), axis=-1, keepdims=True)
        x = x * ops.rsqrt(variance + self.epsilon)
        x = ops.cast(x, self.compute_dtype)
        gamma = ops.cast(self.gamma, self.compute_dtype)
        beta = ops.cast(self.beta, self.compute_dtype)
        return x * gamma + beta

    def compute_output_shape(self, input_shape):
        return input_shape


def _get_norm(norm_type, channels, dtype):
    if norm_type == "rms_norm":
        return RMSNorm2D(channels, dtype=dtype, name="norm")
    elif norm_type == "batch_norm":
        # `epsilon` matches the PyTorch `BatchNorm2d` default of 1e-5.
        return keras.layers.BatchNormalization(
            axis=-1, epsilon=1e-5, momentum=0.9, dtype=dtype, name="norm"
        )
    raise ValueError(f"Unsupported norm_type={norm_type}")


class DCResBlock(keras.layers.Layer):
    """The `ResBlock` used by DC-AE (two 3x3 convs + norm + residual)."""

    def __init__(
        self, channels, norm_type="batch_norm", act_fn="relu", **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.norm_type = norm_type
        self.act_fn = act_fn
        self.activation = keras.activations.get(act_fn)
        self.conv1 = keras.layers.Conv2D(
            self.channels,
            3,
            padding="same",
            dtype=self.dtype_policy,
            name="conv1",
        )
        self.conv2 = keras.layers.Conv2D(
            self.channels,
            3,
            padding="same",
            use_bias=False,
            dtype=self.dtype_policy,
            name="conv2",
        )
        self.norm = _get_norm(norm_type, self.channels, self.dtype_policy)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm(x, training=training)
        return x + inputs

    def compute_output_shape(self, input_shape):
        return input_shape


class GLUMBConv(keras.layers.Layer):
    """Gated linear MBConv used inside `EfficientViTBlock`."""

    def __init__(
        self,
        in_channels,
        out_channels,
        expand_ratio=4,
        norm_type="rms_norm",
        residual_connection=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.expand_ratio = expand_ratio
        self.norm_type = norm_type
        self.residual_connection = residual_connection
        hidden_channels = int(expand_ratio * in_channels)
        self.hidden_channels = hidden_channels

        self.nonlinearity = keras.activations.silu
        self.conv_inverted = keras.layers.Conv2D(
            hidden_channels * 2,
            1,
            dtype=self.dtype_policy,
            name="conv_inverted",
        )
        self.conv_depth = keras.layers.DepthwiseConv2D(
            3,
            padding="same",
            dtype=self.dtype_policy,
            name="conv_depth",
        )
        self.conv_point = keras.layers.Conv2D(
            self.out_channels,
            1,
            use_bias=False,
            dtype=self.dtype_policy,
            name="conv_point",
        )
        self.norm = None
        if norm_type == "rms_norm":
            self.norm = RMSNorm2D(
                self.out_channels, dtype=self.dtype_policy, name="norm"
            )

    def call(self, inputs):
        residual = inputs
        x = self.conv_inverted(inputs)
        x = self.nonlinearity(x)
        x = self.conv_depth(x)
        x, gate = ops.split(x, 2, axis=-1)
        x = x * self.nonlinearity(gate)
        x = self.conv_point(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.residual_connection:
            x = x + residual
        return x

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.out_channels
        return tuple(output_shape)


class SanaMultiscaleLinearAttention(keras.layers.Layer):
    """Lightweight (multi-scale) ReLU linear attention from SANA / DC-AE.

    `qkv_multiscales` is empty for the `dc-ae-f32c32` presets, so only the
    single-scale path is exercised, but the multi-scale projections are built
    when requested for completeness.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        attention_head_dim=32,
        mult=1.0,
        norm_type="rms_norm",
        kernel_sizes=(),
        eps=1e-15,
        residual_connection=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.attention_head_dim = int(attention_head_dim)
        self.mult = mult
        self.norm_type = norm_type
        self.kernel_sizes = tuple(kernel_sizes)
        self.eps = eps
        self.residual_connection = residual_connection

        num_heads = int(in_channels // attention_head_dim * mult)
        self.num_heads = num_heads
        inner_dim = num_heads * self.attention_head_dim
        self.inner_dim = inner_dim

        self.to_q = keras.layers.Dense(
            inner_dim, use_bias=False, dtype=self.dtype_policy, name="to_q"
        )
        self.to_k = keras.layers.Dense(
            inner_dim, use_bias=False, dtype=self.dtype_policy, name="to_k"
        )
        self.to_v = keras.layers.Dense(
            inner_dim, use_bias=False, dtype=self.dtype_policy, name="to_v"
        )
        self.multiscale_projections = []
        for i, kernel_size in enumerate(self.kernel_sizes):
            self.multiscale_projections.append(
                _SanaMultiscaleProjection(
                    inner_dim,
                    num_heads,
                    kernel_size,
                    dtype=self.dtype_policy,
                    name=f"to_qkv_multiscale_{i}",
                )
            )
        self.nonlinearity = keras.activations.relu
        self.to_out = keras.layers.Dense(
            self.out_channels,
            use_bias=False,
            dtype=self.dtype_policy,
            name="to_out",
        )
        self.norm_out = _get_norm(
            norm_type, self.out_channels, self.dtype_policy
        )

    def _linear_attention(self, query, key, value):
        # query/key/value: (B, heads, head_dim, N).
        ones = ops.ones_like(value[:, :, :1, :])
        value = ops.concatenate([value, ones], axis=2)
        scores = ops.matmul(value, ops.transpose(key, (0, 1, 3, 2)))
        hidden = ops.matmul(scores, query)
        hidden = ops.cast(hidden, "float32")
        hidden = hidden[:, :, :-1, :] / (hidden[:, :, -1:, :] + self.eps)
        return ops.cast(hidden, self.compute_dtype)

    def _quadratic_attention(self, query, key, value):
        scores = ops.matmul(ops.transpose(key, (0, 1, 3, 2)), query)
        scores = ops.cast(scores, "float32")
        scores = scores / (ops.sum(scores, axis=2, keepdims=True) + self.eps)
        scores = ops.cast(scores, self.compute_dtype)
        return ops.matmul(value, scores)

    def call(self, inputs):
        residual = inputs
        shape = ops.shape(inputs)
        b, height, width = shape[0], shape[1], shape[2]
        n = height * width
        head_dim = self.attention_head_dim
        # Static dims (concrete at eager call time) decide linear vs quadratic
        # attention; diffusers uses quadratic only when `H * W <= head_dim`.
        static_h, static_w = inputs.shape[1], inputs.shape[2]

        query = self.to_q(inputs)
        key = self.to_k(inputs)
        value = self.to_v(inputs)
        qkv = ops.concatenate([query, key, value], axis=-1)

        multi_scale_qkv = [qkv]
        for projection in self.multiscale_projections:
            multi_scale_qkv.append(projection(qkv))
        qkv = ops.concatenate(multi_scale_qkv, axis=-1)

        # Each multiscale projection appends another `3 * inner_dim` block, so
        # the base heads are repeated `1 + num_scales` times (diffusers treats
        # the appended scales as additional heads).
        total_heads = self.num_heads * len(multi_scale_qkv)
        total_inner = self.inner_dim * len(multi_scale_qkv)

        # Mimic the channels-first reshape in diffusers:
        # (B, H, W, C_total) -> (B, C_total, N) -> (B, heads, 3 * head_dim, N).
        qkv = ops.reshape(qkv, (b, n, qkv.shape[-1]))
        qkv = ops.transpose(qkv, (0, 2, 1))
        qkv = ops.reshape(qkv, (b, total_heads, 3 * head_dim, n))
        query, key, value = ops.split(qkv, 3, axis=2)
        query = self.nonlinearity(query)
        key = self.nonlinearity(key)

        if static_h is None or static_w is None:
            use_linear = True
        else:
            use_linear = static_h * static_w > head_dim
        if use_linear:
            hidden = self._linear_attention(query, key, value)
        else:
            hidden = self._quadratic_attention(query, key, value)

        hidden = ops.reshape(hidden, (b, total_inner, n))
        hidden = ops.transpose(hidden, (0, 2, 1))
        hidden = ops.reshape(hidden, (b, height, width, total_inner))

        hidden = self.to_out(hidden)
        hidden = self.norm_out(hidden)
        if self.residual_connection:
            hidden = hidden + residual
        return hidden

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.out_channels
        return tuple(output_shape)


class _SanaMultiscaleProjection(keras.layers.Layer):
    """Depthwise multi-scale QKV aggregation.

    Implements diffusers' `SanaMultiscaleAttentionProjection`.
    """

    def __init__(self, inner_dim, num_heads, kernel_size, **kwargs):
        super().__init__(**kwargs)
        channels = 3 * inner_dim
        self.proj_in = keras.layers.DepthwiseConv2D(
            kernel_size,
            padding="same",
            use_bias=False,
            dtype=self.dtype_policy,
            name="proj_in",
        )
        self.proj_out = keras.layers.Conv2D(
            channels,
            1,
            groups=3 * num_heads,
            use_bias=False,
            dtype=self.dtype_policy,
            name="proj_out",
        )

    def call(self, inputs):
        x = self.proj_in(inputs)
        x = self.proj_out(x)
        return x


class EfficientViTBlock(keras.layers.Layer):
    """Sana multi-scale linear attention followed by a `GLUMBConv`."""

    def __init__(
        self,
        channels,
        attention_head_dim=32,
        norm_type="rms_norm",
        qkv_multiscales=(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.attn = SanaMultiscaleLinearAttention(
            in_channels=channels,
            out_channels=channels,
            attention_head_dim=attention_head_dim,
            norm_type=norm_type,
            kernel_sizes=qkv_multiscales,
            residual_connection=True,
            dtype=self.dtype_policy,
            name="attn",
        )
        self.conv_out = GLUMBConv(
            in_channels=channels,
            out_channels=channels,
            norm_type="rms_norm",
            dtype=self.dtype_policy,
            name="conv_out",
        )

    def call(self, inputs):
        x = self.attn(inputs)
        x = self.conv_out(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class DCDownBlock2d(keras.layers.Layer):
    """Residual `pixel_unshuffle` downsample ("residual autoencoding")."""

    def __init__(
        self,
        in_channels,
        out_channels,
        downsample=True,
        shortcut=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.downsample = downsample
        self.shortcut = shortcut
        self.factor = 2
        self.group_size = self.in_channels * self.factor**2 // self.out_channels
        stride = 1 if downsample else 2
        conv_out_channels = (
            self.out_channels // self.factor**2
            if downsample
            else self.out_channels
        )
        self.conv = keras.layers.Conv2D(
            conv_out_channels,
            3,
            strides=stride,
            padding="same",
            dtype=self.dtype_policy,
            name="conv",
        )

    def call(self, inputs):
        x = self.conv(inputs)
        if self.downsample:
            x = pixel_unshuffle(x, self.factor)
        if self.shortcut:
            y = pixel_unshuffle(inputs, self.factor)
            shape = ops.shape(y)
            b, h, w = shape[0], shape[1], shape[2]
            y = ops.reshape(y, (b, h, w, self.out_channels, self.group_size))
            y = ops.mean(y, axis=-1)
            return x + y
        return x

    def compute_output_shape(self, input_shape):
        # Spatial is halved either way: the `pixel_unshuffle` path
        # (`downsample=True`) unshuffles by `factor`, while the `Conv` path
        # (`downsample=False`) uses a stride-`factor` conv.
        b, h, w, _ = input_shape
        h = None if h is None else h // self.factor
        w = None if w is None else w // self.factor
        return (b, h, w, self.out_channels)


class DCUpBlock2d(keras.layers.Layer):
    """Residual `pixel_shuffle` upsample ("residual autoencoding")."""

    def __init__(self, in_channels, out_channels, shortcut=True, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.shortcut = shortcut
        self.factor = 2
        self.repeats = self.out_channels * self.factor**2 // self.in_channels
        self.conv = keras.layers.Conv2D(
            self.out_channels * self.factor**2,
            3,
            padding="same",
            dtype=self.dtype_policy,
            name="conv",
        )

    def call(self, inputs):
        x = self.conv(inputs)
        x = pixel_shuffle(x, self.factor)
        if self.shortcut:
            y = ops.repeat(inputs, self.repeats, axis=-1)
            y = pixel_shuffle(y, self.factor)
            return x + y
        return x

    def compute_output_shape(self, input_shape):
        b, h, w, _ = input_shape
        h = None if h is None else h * self.factor
        w = None if w is None else w * self.factor
        return (b, h, w, self.out_channels)


def _get_block(
    block_type,
    channels,
    attention_head_dim,
    norm_type,
    act_fn,
    qkv_multiscales,
    dtype,
    name,
):
    if block_type == "ResBlock":
        return DCResBlock(channels, norm_type, act_fn, dtype=dtype, name=name)
    elif block_type == "EfficientViTBlock":
        return EfficientViTBlock(
            channels,
            attention_head_dim=attention_head_dim,
            norm_type=norm_type,
            qkv_multiscales=qkv_multiscales,
            dtype=dtype,
            name=name,
        )
    raise ValueError(f"Unsupported block_type={block_type}")


class DCAEEncoder(keras.layers.Layer):
    """The DC-AE encoder: image -> deterministic latent."""

    def __init__(
        self,
        in_channels,
        latent_channels,
        attention_head_dim,
        block_types,
        block_out_channels,
        layers_per_block,
        qkv_multiscales,
        downsample_block_type="pixel_unshuffle",
        out_shortcut=True,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        data_format = standardize_data_format(data_format)
        if data_format != "channels_last":
            raise NotImplementedError("DC-AE only supports channels_last.")
        num_blocks = len(block_out_channels)
        downsample = downsample_block_type == "pixel_unshuffle"
        first_channels = (
            block_out_channels[0]
            if layers_per_block[0] > 0
            else block_out_channels[1]
        )

        if layers_per_block[0] > 0:
            self.conv_in = keras.layers.Conv2D(
                first_channels,
                3,
                padding="same",
                dtype=self.dtype_policy,
                name="conv_in",
            )
        else:
            self.conv_in = DCDownBlock2d(
                in_channels,
                first_channels,
                downsample=downsample,
                shortcut=False,
                dtype=self.dtype_policy,
                name="conv_in",
            )

        # The encoder is purely sequential, so the blocks are kept in a single
        # flat list (a list-of-lists does not round-trip through Keras
        # serialization). `block_keys` records the diffusers weight prefix of
        # each block for the checkpoint converter.
        self.blocks = []
        self.block_keys = []
        for i in range(num_blocks):
            out_channel = block_out_channels[i]
            num_layers = layers_per_block[i]
            for j in range(num_layers):
                self.blocks.append(
                    _get_block(
                        block_types[i],
                        out_channel,
                        attention_head_dim,
                        norm_type="rms_norm",
                        act_fn="silu",
                        qkv_multiscales=qkv_multiscales[i],
                        dtype=self.dtype_policy,
                        name=f"down_blocks_{i}_{j}",
                    )
                )
                self.block_keys.append(f"encoder.down_blocks.{i}.{j}")
            if i < num_blocks - 1 and num_layers > 0:
                self.blocks.append(
                    DCDownBlock2d(
                        out_channel,
                        block_out_channels[i + 1],
                        downsample=downsample,
                        shortcut=True,
                        dtype=self.dtype_policy,
                        name=f"down_blocks_{i}_{num_layers}",
                    )
                )
                self.block_keys.append(f"encoder.down_blocks.{i}.{num_layers}")

        self.conv_out = keras.layers.Conv2D(
            latent_channels,
            3,
            padding="same",
            dtype=self.dtype_policy,
            name="conv_out",
        )
        self.out_shortcut = out_shortcut
        self.latent_channels = latent_channels
        self.out_shortcut_group_size = block_out_channels[-1] // latent_channels

    def call(self, inputs):
        x = self.conv_in(inputs)
        for block in self.blocks:
            x = block(x)
        if self.out_shortcut:
            shape = ops.shape(x)
            b, h, w = shape[0], shape[1], shape[2]
            shortcut = ops.reshape(
                x, (b, h, w, self.latent_channels, self.out_shortcut_group_size)
            )
            shortcut = ops.mean(shortcut, axis=-1)
            return self.conv_out(x) + shortcut
        return self.conv_out(x)

    def compute_output_shape(self, input_shape):
        # Chain the static shapes of the sublayers so the functional build
        # stays symbolic. Without this, Keras cannot infer the output shape
        # (the dynamic `ops.shape` calls collapse it to `None`) and the torch
        # backend falls back to eagerly running the encoder on the GPU.
        shape = self.conv_in.compute_output_shape(input_shape)
        for block in self.blocks:
            shape = block.compute_output_shape(shape)
        shape = list(self.conv_out.compute_output_shape(shape))
        shape[-1] = self.latent_channels
        return tuple(shape)


class DCAEDecoder(keras.layers.Layer):
    """The DC-AE decoder: latent -> reconstructed image."""

    def __init__(
        self,
        in_channels,
        latent_channels,
        attention_head_dim,
        block_types,
        block_out_channels,
        layers_per_block,
        qkv_multiscales,
        norm_types,
        act_fns,
        upsample_block_type="pixel_shuffle",
        in_shortcut=True,
        conv_act_fn="relu",
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        data_format = standardize_data_format(data_format)
        if data_format != "channels_last":
            raise NotImplementedError("DC-AE only supports channels_last.")
        num_blocks = len(block_out_channels)

        self.conv_in = keras.layers.Conv2D(
            block_out_channels[-1],
            3,
            padding="same",
            dtype=self.dtype_policy,
            name="conv_in",
        )
        self.in_shortcut = in_shortcut
        self.in_shortcut_repeats = block_out_channels[-1] // latent_channels

        # The decoder runs the stages from the deepest (i = num_blocks - 1)
        # down to stage 0. The blocks are stored in a single flat list in
        # execution order (a list-of-lists does not round-trip through Keras
        # serialization). `block_keys` records the diffusers weight prefix of
        # each block for the checkpoint converter.
        self.blocks = []
        self.block_keys = []
        for i in reversed(range(num_blocks)):
            out_channel = block_out_channels[i]
            num_layers = layers_per_block[i]
            offset = 0
            if i < num_blocks - 1 and num_layers > 0:
                self.blocks.append(
                    DCUpBlock2d(
                        block_out_channels[i + 1],
                        out_channel,
                        shortcut=True,
                        dtype=self.dtype_policy,
                        name=f"up_blocks_{i}_0",
                    )
                )
                self.block_keys.append(f"decoder.up_blocks.{i}.0")
                offset = 1
            for j in range(num_layers):
                self.blocks.append(
                    _get_block(
                        block_types[i],
                        out_channel,
                        attention_head_dim,
                        norm_type=norm_types[i],
                        act_fn=act_fns[i],
                        qkv_multiscales=qkv_multiscales[i],
                        dtype=self.dtype_policy,
                        name=f"up_blocks_{i}_{j + offset}",
                    )
                )
                self.block_keys.append(f"decoder.up_blocks.{i}.{j + offset}")

        channels = (
            block_out_channels[0]
            if layers_per_block[0] > 0
            else block_out_channels[1]
        )
        self.norm_out = RMSNorm2D(
            channels, dtype=self.dtype_policy, name="norm_out"
        )
        self.conv_act = keras.activations.get(conv_act_fn)
        if layers_per_block[0] > 0:
            self.conv_out = keras.layers.Conv2D(
                in_channels,
                3,
                padding="same",
                dtype=self.dtype_policy,
                name="conv_out",
            )
        else:
            self.conv_out = DCUpBlock2d(
                channels,
                in_channels,
                shortcut=False,
                dtype=self.dtype_policy,
                name="conv_out",
            )

    def call(self, inputs):
        if self.in_shortcut:
            shortcut = ops.repeat(inputs, self.in_shortcut_repeats, axis=-1)
            x = self.conv_in(inputs) + shortcut
        else:
            x = self.conv_in(inputs)
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x

    def compute_output_shape(self, input_shape):
        # Chain the static shapes of the sublayers so the functional build
        # stays symbolic. Without this, Keras cannot infer the output shape
        # (the dynamic `ops.shape` calls collapse it to `None`) and the torch
        # backend falls back to eagerly running the full decoder on the GPU,
        # which exhausts memory.
        shape = self.conv_in.compute_output_shape(input_shape)
        for block in self.blocks:
            shape = block.compute_output_shape(shape)
        shape = self.norm_out.compute_output_shape(shape)
        shape = self.conv_out.compute_output_shape(shape)
        return shape
