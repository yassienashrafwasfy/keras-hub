import keras
from keras import backend
from keras import distribution
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.stable_diffusion_1_4.dpm_solver_multistep_scheduler import (  # noqa: E501
    DPMSolverMultistepScheduler,
)
from keras_hub.src.models.stable_diffusion_1_4.unet import UNet
from keras_hub.src.utils.keras_utils import standardize_data_format


class ImageRescaling(layers.Rescaling):
    """Rescales image-space latents into the diffusion latent space."""

    def call(self, inputs):
        dtype = self.compute_dtype
        scale = self.backend.cast(self.scale, dtype)
        offset = self.backend.cast(self.offset, dtype)
        return (self.backend.cast(inputs, dtype) - offset) * scale


class LatentRescaling(layers.Rescaling):
    """Rescales diffusion latents back into the VAE latent space."""

    def call(self, inputs):
        dtype = self.compute_dtype
        scale = self.backend.cast(self.scale, dtype)
        offset = self.backend.cast(self.offset, dtype)
        return (self.backend.cast(inputs, dtype) / scale) + offset


class TimestepBroadcastTo(layers.Layer):
    """Broadcast a scalar timestep to the batch dimension of the latents."""

    def call(self, latents, timestep):
        return ops.broadcast_to(timestep, ops.shape(latents)[:1])


class ClassifierFreeGuidance(layers.Layer):
    """Combine the positive and negative noise predictions.

    Call arguments:
        inputs: A concatenation of the positive and negative noise residuals
            along the batch axis.
        guidance_scale: The classifier-free guidance scale.

    Reference:
    - [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
    """

    def call(self, inputs, guidance_scale):
        positive_noise, negative_noise = ops.split(inputs, 2, axis=0)
        return ops.add(
            negative_noise,
            ops.multiply(
                guidance_scale, ops.subtract(positive_noise, negative_noise)
            ),
        )

    def compute_output_shape(self, inputs_shape):
        outputs_shape = list(inputs_shape)
        if outputs_shape[0] is not None:
            outputs_shape[0] = outputs_shape[0] // 2
        return outputs_shape


@keras_hub_export("keras_hub.models.StableDiffusion1_4Backbone")
class StableDiffusion1_4Backbone(Backbone):
    """Stable Diffusion v1.4 core network with hyperparameters.

    This backbone imports a CLIP model as the text encoder and implements the
    base UNet and VAE networks for the Stable Diffusion v1.4 model. The noise
    schedule is handled by a DPM-Solver++ (2M) scheduler with Karras sigmas.

    The default constructor gives a fully customizable, randomly initialized
    model with any hyperparameters. To load preset architectures and weights,
    use the `from_preset` constructor.

    Args:
        unet_in_channels: int. The number of channels in the latent input to
            the UNet.
        unet_block_out_channels: list of ints. The number of filters in each
            UNet down/up block.
        unet_layers_per_block: int. The number of ResNet layers per UNet block.
        unet_num_heads: int. The number of attention heads in each UNet
            transformer.
        clip: A `keras_hub.models.CLIPTextEncoder` instance.
        vae: A `keras_hub.models.VAEBackbone` instance.
        num_train_timesteps: int. The number of diffusion steps used to train
            the model. Defaults to `1000`.
        latent_channels: int. The number of channels in the latent. Defaults to
            `4`.
        output_channels: int. The number of channels in the output. Defaults to
            `3`.
        image_shape: tuple. The input shape without the batch size. Defaults to
            `(512, 512, 3)`.
        data_format: `None` or str. Only `"channels_last"` is supported.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the model's computations and weights.

    Example:
    ```python
    # Pretrained Stable Diffusion v1.4 model.
    model = keras_hub.models.StableDiffusion1_4Backbone.from_preset(
        "stable_diffusion_1_4"
    )

    # Randomly initialized model with custom config.
    clip = keras_hub.models.CLIPTextEncoder(...)
    vae = keras_hub.models.VAEBackbone(...)
    model = keras_hub.models.StableDiffusion1_4Backbone(
        unet_block_out_channels=[320, 640, 1280, 1280],
        unet_layers_per_block=2,
        unet_num_heads=8,
        clip=clip,
        vae=vae,
    )
    ```
    """

    def __init__(
        self,
        clip,
        vae,
        unet_in_channels=4,
        unet_block_out_channels=(320, 640, 1280, 1280),
        unet_layers_per_block=2,
        unet_num_heads=8,
        num_train_timesteps=1000,
        latent_channels=4,
        output_channels=3,
        image_shape=(512, 512, 3),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)
        if data_format != "channels_last":
            raise NotImplementedError
        height, width = image_shape[0], image_shape[1]
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                "height and width in `image_shape` must be divisible by 8. "
                f"Received: image_shape={image_shape}"
            )
        latent_shape = (height // 8, width // 8, int(latent_channels))
        self._latent_shape = latent_shape

        # === Layers ===
        self.clip = clip
        self.vae = vae
        self.diffuser = UNet(
            in_channels=unet_in_channels,
            out_channels=latent_channels,
            block_out_channels=unet_block_out_channels,
            layers_per_block=unet_layers_per_block,
            num_heads=unet_num_heads,
            cross_attention_dim=clip.hidden_dim,
            context_max_length=clip.max_sequence_length,
            data_format=data_format,
            dtype=dtype,
            name="diffuser",
        )
        self.cfg = ClassifierFreeGuidance(
            dtype=dtype, name="classifier_free_guidance"
        )
        self.timestep_broadcast_to = TimestepBroadcastTo(
            dtype=dtype, name="timestep_broadcast_to"
        )
        # Set `dtype="float32"` to keep the scheduler math in high precision.
        self.scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            dtype="float32",
            name="scheduler",
        )
        self.image_rescaling = ImageRescaling(
            scale=self.vae.scale,
            offset=self.vae.shift,
            dtype=dtype,
            name="image_rescaling",
        )
        self.latent_rescaling = LatentRescaling(
            scale=self.vae.scale,
            offset=self.vae.shift,
            dtype=dtype,
            name="latent_rescaling",
        )
        # 1x1 convs that map between the VAE moments/latent and the diffusion
        # latent, matching diffusers' `AutoencoderKL.quant_conv` and
        # `post_quant_conv`.
        self.quant_conv = layers.Conv2D(
            2 * int(latent_channels), 1, dtype=dtype, name="quant_conv"
        )
        self.post_quant_conv = layers.Conv2D(
            int(latent_channels), 1, dtype=dtype, name="post_quant_conv"
        )

        # === Functional Model ===
        image_input = keras.Input(shape=image_shape, name="images")
        latent_input = keras.Input(shape=latent_shape, name="latents")
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        negative_token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="negative_token_ids"
        )
        num_step_input = keras.Input(shape=(), dtype="int32", name="num_steps")
        guidance_scale_input = keras.Input(
            shape=(), dtype="float32", name="guidance_scale"
        )

        embeddings = self.encode_text_step(
            token_id_input, negative_token_id_input
        )
        latents = self.encode_image_step(image_input)
        # Use `steps=0` to define the functional model.
        denoised_latents, _ = self.denoise_step(
            latent_input,
            ops.zeros_like(latent_input),
            embeddings,
            0,
            num_step_input[0],
            guidance_scale_input[0],
        )
        images = self.decode_step(denoised_latents)

        super().__init__(
            inputs={
                "images": image_input,
                "latents": latent_input,
                "token_ids": token_id_input,
                "negative_token_ids": negative_token_id_input,
                "num_steps": num_step_input,
                "guidance_scale": guidance_scale_input,
            },
            outputs={"latents": latents, "images": images},
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.unet_in_channels = unet_in_channels
        self.unet_block_out_channels = unet_block_out_channels
        self.unet_layers_per_block = unet_layers_per_block
        self.unet_num_heads = unet_num_heads
        self.num_train_timesteps = num_train_timesteps
        self.latent_channels = latent_channels
        self.output_channels = output_channels
        self.image_shape = image_shape

    @property
    def latent_shape(self):
        return (None,) + self._latent_shape

    def encode_text_step(self, token_ids, negative_token_ids):
        def encode(token_ids):
            return self.clip({"token_ids": token_ids}, training=False)

        return encode(token_ids), encode(negative_token_ids)

    def encode_image_step(self, images):
        # `quant_conv` acts on the moments, before the gaussian sampler.
        moments = self.vae.encoder(images)
        moments = self.quant_conv(moments)
        latents = self.vae.distribution_sampler(moments)
        return self.image_rescaling(latents)

    def configure_scheduler(self, num_steps):
        self.scheduler.set_timesteps(num_steps)

    def denoise_step(
        self,
        latents,
        prev_x0,
        embeddings,
        step,
        num_steps,
        guidance_scale=None,
    ):
        step = ops.convert_to_tensor(step, dtype="int32")
        timestep = ops.take(self.scheduler.timesteps, step)

        if guidance_scale is not None and not keras.utils.is_keras_tensor(
            guidance_scale
        ):
            guidance_scale = ops.convert_to_tensor(guidance_scale)

        if guidance_scale is not None:
            latent_model_input = ops.concatenate([latents, latents], axis=0)
            context = ops.concatenate([embeddings[0], embeddings[1]], axis=0)
        else:
            latent_model_input = latents
            context = embeddings[0]
        timesteps = self.timestep_broadcast_to(latent_model_input, timestep)

        noise_residual = self.diffuser(
            {
                "latent": latent_model_input,
                "timestep": timesteps,
                "context": context,
            },
            training=False,
        )
        if guidance_scale is not None:
            noise_residual = self.cfg(noise_residual, guidance_scale)

        x0 = self.scheduler.convert_model_output(noise_residual, step, latents)
        prev_sample = self.scheduler.step(x0, prev_x0, step, latents)
        return prev_sample, x0

    def decode_step(self, latents):
        latents = self.latent_rescaling(latents)
        latents = self.post_quant_conv(latents)
        return self.vae.decoder(latents, training=False)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "unet_in_channels": self.unet_in_channels,
                "unet_block_out_channels": self.unet_block_out_channels,
                "unet_layers_per_block": self.unet_layers_per_block,
                "unet_num_heads": self.unet_num_heads,
                "clip": layers.serialize(self.clip),
                "vae": layers.serialize(self.vae),
                "num_train_timesteps": self.num_train_timesteps,
                "latent_channels": self.latent_channels,
                "output_channels": self.output_channels,
                "image_shape": self.image_shape,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()

        # Propagate `dtype` to the VAE if needed.
        if "dtype" in config and config["dtype"] is not None:
            if "dtype" not in config["vae"]["config"]:
                config["vae"]["config"]["dtype"] = config["dtype"]

        # The text encoder defaults to float16 unless on JAX CPU.
        # TODO: JAX CPU doesn't support float16 in `nn.dot_product_attention`.
        is_jax_cpu = (
            backend.backend() == "jax"
            and "cpu" in distribution.list_devices()[0].lower()
        )
        if (
            config["clip"] is not None
            and "dtype" not in config["clip"]["config"]
            and not is_jax_cpu
        ):
            config["clip"]["config"]["dtype"] = "float16"

        # We expect `clip` and `vae` to be instantiated.
        config["clip"] = layers.deserialize(
            config["clip"], custom_objects=custom_objects
        )
        config["vae"] = layers.deserialize(
            config["vae"], custom_objects=custom_objects
        )
        return cls(**config)
