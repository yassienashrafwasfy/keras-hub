import math

from keras import layers
from keras import ops

from keras_hub.src.utils.keras_utils import standardize_data_format


def quick_gelu(x):
    return x * ops.sigmoid(1.702 * x)


class MobileCLIP2ConvStem(layers.Layer):
    """The convolutional patch-embedding stem of MobileCLIP2-B.

    This replaces the single-Conv2D patchify used by standard CLIP with the
    `ConvStem` (a.k.a. `mci`) used by timm's `vit_base_mci_224`: a stack of
    `Conv2D -> BatchNorm -> GELU` blocks (the last block is a bare `Conv2D`
    with bias and no norm/activation). The spatial feature map is flattened
    into a sequence of patch tokens.

    Args:
        hidden_dim: int. The output channel dimension (patch token dim).
        channels: tuple of int. The output channels of each conv layer.
        kernel_sizes: tuple of int. The kernel size of each conv layer.
        strides: tuple of int. The stride of each conv layer.
    """

    def __init__(
        self,
        hidden_dim,
        channels=(192, 192, 768),
        kernel_sizes=(4, 2, 2),
        strides=(4, 2, 2),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = int(hidden_dim)
        self.channels = tuple(channels)
        self.kernel_sizes = tuple(kernel_sizes)
        self.strides = tuple(strides)
        data_format = standardize_data_format(data_format)
        self.data_format = data_format
        bn_axis = -1 if data_format == "channels_last" else 1

        self.convs = []
        self.norms = []
        self.activations = []
        for i in range(len(self.channels)):
            last_conv = i == len(self.channels) - 1
            self.convs.append(
                layers.Conv2D(
                    self.channels[i],
                    kernel_size=self.kernel_sizes[i],
                    strides=self.strides[i],
                    padding="valid",
                    data_format=data_format,
                    use_bias=last_conv,
                    dtype=dtype,
                    name=f"conv_{i}",
                )
            )
            if not last_conv:
                self.norms.append(
                    layers.BatchNormalization(
                        axis=bn_axis,
                        epsilon=1e-5,  # timm `BatchNorm2d` default.
                        dtype=dtype,
                        name=f"batch_norm_{i}",
                    )
                )
                self.activations.append(
                    layers.Activation("gelu", dtype=dtype, name=f"gelu_{i}")
                )
            else:
                self.norms.append(None)
                self.activations.append(None)

    def build(self, input_shape):
        shape = input_shape
        for i, conv in enumerate(self.convs):
            conv.build(shape)
            shape = conv.compute_output_shape(shape)
            if self.norms[i] is not None:
                self.norms[i].build(shape)
        self.built = True

    def call(self, inputs, training=None):
        x = inputs
        for conv, norm, activation in zip(
            self.convs, self.norms, self.activations
        ):
            x = conv(x, training=training)
            if norm is not None:
                x = norm(x, training=training)
                x = activation(x)
        batch_size = ops.shape(x)[0]
        if self.data_format == "channels_last":
            x = ops.reshape(x, (batch_size, -1, self.hidden_dim))
        else:
            x = ops.reshape(x, (batch_size, self.hidden_dim, -1))
            x = ops.transpose(x, (0, 2, 1))
        return x

    def compute_output_shape(self, input_shape):
        shape = input_shape
        for conv in self.convs:
            shape = conv.compute_output_shape(shape)
        if self.data_format == "channels_last":
            spatial = shape[1:3]
        else:
            spatial = shape[2:4]
        if spatial[0] is not None and spatial[1] is not None:
            num_patches = spatial[0] * spatial[1]
        else:
            num_patches = None
        return (input_shape[0], num_patches, self.hidden_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "channels": self.channels,
                "kernel_sizes": self.kernel_sizes,
                "strides": self.strides,
            }
        )
        return config


class MobileCLIP2VisionEmbedding(layers.Layer):
    """The vision embedding of MobileCLIP2-B.

    Runs the `MobileCLIP2ConvStem`, adds a patch-only position embedding
    (matching timm's `no_embed_class=True`, so the class token receives no
    positional embedding), then prepends the learnable class token.

    Args:
        hidden_dim: int. The patch token dimension.
        image_size: int. The (square) input image size.
        stem_channels: tuple of int. The `MobileCLIP2ConvStem` channels.
        stem_kernel_sizes: tuple of int. The `MobileCLIP2ConvStem` kernels.
        stem_strides: tuple of int. The `MobileCLIP2ConvStem` strides.
    """

    def __init__(
        self,
        hidden_dim,
        image_size,
        stem_channels=(192, 192, 768),
        stem_kernel_sizes=(4, 2, 2),
        stem_strides=(4, 2, 2),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = int(hidden_dim)
        self.image_size = int(image_size)
        self.stem_channels = tuple(stem_channels)
        self.stem_kernel_sizes = tuple(stem_kernel_sizes)
        self.stem_strides = tuple(stem_strides)
        data_format = standardize_data_format(data_format)
        self.data_format = data_format

        total_stride = 1
        for stride in self.stem_strides:
            total_stride *= stride
        self.num_patches = (self.image_size // total_stride) ** 2

        self.conv_stem = MobileCLIP2ConvStem(
            hidden_dim=hidden_dim,
            channels=stem_channels,
            kernel_sizes=stem_kernel_sizes,
            strides=stem_strides,
            data_format=data_format,
            dtype=dtype,
            name="conv_stem",
        )
        self.position_embedding = layers.Embedding(
            self.num_patches,
            hidden_dim,
            dtype=dtype,
            name="position_embedding",
        )

    def build(self, input_shape):
        self.class_embedding = self.add_weight(
            shape=(self.hidden_dim,),
            initializer="random_normal",
            dtype=self.variable_dtype,
            name="class_embedding",
        )
        self.position_ids = self.add_weight(
            shape=(1, self.num_patches),
            initializer="zeros",
            # Let the backend determine the int dtype. For example, tf
            # requires int64 for correct device placement, whereas jax and torch
            # don't.
            dtype=int,
            trainable=False,
            name="position_ids",
        )
        self.conv_stem.build(input_shape)
        self.position_embedding.build(self.position_ids.shape)
        self.built = True

    def call(self, inputs, training=None):
        x = inputs
        batch_size = ops.shape(x)[0]
        patch_embeddings = self.conv_stem(x, training=training)
        position_embeddings = self.position_embedding(self.position_ids)
        patch_embeddings = ops.add(patch_embeddings, position_embeddings)
        class_embeddings = ops.expand_dims(self.class_embedding, axis=(0, 1))
        class_embeddings = ops.tile(class_embeddings, (batch_size, 1, 1))
        return ops.concatenate([class_embeddings, patch_embeddings], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_patches + 1, self.hidden_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "image_size": self.image_size,
                "stem_channels": self.stem_channels,
                "stem_kernel_sizes": self.stem_kernel_sizes,
                "stem_strides": self.stem_strides,
            }
        )
        return config


class MobileCLIP2EncoderLayer(layers.Layer):
    """A pre-norm transformer block shared by both MobileCLIP2 towers.

    Identical to CLIP's encoder layer, but exposes `layer_norm_epsilon`
    (timm's vision tower uses `1e-6`, open_clip's text tower uses `1e-5`) and
    defaults to plain `gelu` activation.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        intermediate_dim,
        intermediate_activation="gelu",
        layer_norm_epsilon=1e-6,
        use_causal_mask=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "`hidden_dim` must be divisible by `num_heads`. "
                f"Received: hidden_dim={hidden_dim}, num_heads={num_heads}"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.intermediate_activation = intermediate_activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_causal_mask = use_causal_mask

        if intermediate_activation == "quick_gelu":
            intermediate_activation = quick_gelu

        self.layer_norm_1 = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm_1",
        )
        self.attention = layers.MultiHeadAttention(
            num_heads,
            hidden_dim // num_heads,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.layer_norm_2 = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm_2",
        )
        self.dense_1 = layers.Dense(
            self.intermediate_dim, dtype=self.dtype_policy, name="dense_1"
        )
        self.activation = layers.Activation(
            intermediate_activation, dtype=self.dtype_policy, name="activation"
        )
        self.dense_2 = layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="dense_2"
        )

    def build(self, input_shape):
        self.layer_norm_1.build(input_shape)
        self.attention.build(input_shape, input_shape, input_shape)
        self.layer_norm_2.build(input_shape)
        self.dense_1.build(input_shape)
        input_shape = self.dense_1.compute_output_shape(input_shape)
        self.dense_2.build(input_shape)

    def compute_output_shape(self, inputs_shape):
        outputs_shape = list(inputs_shape)
        outputs_shape[-1] = self.hidden_dim
        return outputs_shape

    def call(self, x, training=None):
        residual = x
        x = self.layer_norm_1(x)
        x = self.attention(
            x, x, x, training=training, use_causal_mask=self.use_causal_mask
        )
        x = ops.add(residual, x)

        residual = x
        x = self.dense_1(self.layer_norm_2(residual))
        x = self.activation(x)
        x = self.dense_2(x)
        x = ops.add(residual, x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "intermediate_activation": self.intermediate_activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "use_causal_mask": self.use_causal_mask,
            }
        )
        return config


class MobileCLIP2VisionPooler(layers.Layer):
    """Extracts the class token (index `0`) as the pooled vision output."""

    def call(self, vision_embeddings):
        return vision_embeddings[:, 0, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class MobileCLIP2TextPooler(layers.Layer):
    """Extracts the text embeddings at the EOS (end-of-text) token positions."""

    def call(self, text_embeddings, token_ids):
        # `keepdims` is not supported in `keras<=3.1`.
        eos_index = ops.argmax(token_ids, axis=-1)
        eos_index = ops.expand_dims(eos_index, axis=-1)
        eos_index = ops.expand_dims(eos_index, axis=-1)
        pooled_outputs = ops.take_along_axis(text_embeddings, eos_index, axis=1)
        return ops.squeeze(pooled_outputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class MobileCLIP2Head(layers.Layer):
    """Computes the contrastive logits from image and text embeddings.

    Both embeddings are L2 normalized and used to compute pairwise cosine
    similarity. The resulting logits are scaled by a learnable `logit_scale`.
    """

    def build(self, input_shape):
        self.logit_scale = self.add_weight(
            shape=(),
            initializer=lambda *a, **kw: math.log(1 / 0.07),
            trainable=True,
            dtype=self.variable_dtype,
            name="logit_scale",
        )

    def call(self, vision_embedding, text_embedding):
        normalized_vision_embedding = ops.sqrt(
            ops.sum(ops.power(vision_embedding, 2), axis=-1, keepdims=True)
        )
        normalized_text_embedding = ops.sqrt(
            ops.sum(ops.power(text_embedding, 2), axis=-1, keepdims=True)
        )
        vision_embedding = vision_embedding / normalized_vision_embedding
        text_embedding = text_embedding / normalized_text_embedding
        logit_scale = ops.exp(self.logit_scale)
        text_logits = (
            ops.matmul(
                text_embedding,
                ops.transpose(vision_embedding),
            )
            * logit_scale
        )
        vision_logits = ops.transpose(text_logits)
        return vision_logits, text_logits

    def compute_output_shape(
        self, vision_embedding_shape, text_embedding_shape
    ):
        vision_logits_shape = (
            vision_embedding_shape[0],
            text_embedding_shape[0],
        )
        text_logits_shape = (
            text_embedding_shape[0],
            vision_embedding_shape[0],
        )
        return vision_logits_shape, text_logits_shape
