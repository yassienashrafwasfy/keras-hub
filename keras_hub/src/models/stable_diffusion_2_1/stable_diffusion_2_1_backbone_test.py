import pytest
from keras import backend
from keras import distribution
from keras import ops

from keras_hub.src.models.clip.clip_text_encoder import CLIPTextEncoder
from keras_hub.src.models.stable_diffusion_2_1.stable_diffusion_2_1_backbone import (  # noqa: E501
    StableDiffusion2_1Backbone,
)
from keras_hub.src.models.vae.vae_backbone import VAEBackbone
from keras_hub.src.tests.test_case import TestCase


class StableDiffusion2_1BackboneTest(TestCase):
    def setUp(self):
        # TODO: JAX CPU doesn't support float16 in `nn.dot_product_attention`.
        is_jax_cpu = (
            backend.backend() == "jax"
            and "cpu" in distribution.list_devices()[0].lower()
        )

        image_shape = (64, 64, 3)
        height, width = image_shape[0], image_shape[1]
        vae = VAEBackbone(
            [32, 32, 32, 32],
            [1, 1, 1, 1],
            [32, 32, 32, 32],
            [1, 1, 1, 1],
            # Use `mode` to generate a deterministic output.
            sampler_method="mode",
            sample_channels=8,
            scale=0.18215,
            shift=0.0,
            name="vae",
        )
        clip = CLIPTextEncoder(
            20,
            64,
            64,
            2,
            2,
            128,
            "gelu",
            dtype="float16" if not is_jax_cpu else None,
            name="clip",
        )
        self.init_kwargs = {
            "clip": clip,
            "vae": vae,
            "unet_block_out_channels": (32, 64),
            "unet_layers_per_block": 1,
            "unet_head_dim": 32,
            "image_shape": image_shape,
        }
        self.input_data = {
            "images": ops.ones((2, height, width, 3)),
            "latents": ops.ones((2, height // 8, width // 8, 4)),
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "negative_token_ids": ops.ones((2, 5), dtype="int32"),
            "num_steps": ops.ones((2,), dtype="int32"),
            "guidance_scale": ops.ones((2,)),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=StableDiffusion2_1Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "images": (2, 64, 64, 3),
                "latents": (2, 8, 8, 4),
            },
            # Since `clip` and `vae` were instantiated outside of
            # `StableDiffusion2_1Backbone`, the mixed precision and
            # quantization checks will fail.
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=StableDiffusion2_1Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
