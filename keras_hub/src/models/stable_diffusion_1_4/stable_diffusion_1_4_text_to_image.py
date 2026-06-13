from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.stable_diffusion_1_4.stable_diffusion_1_4_backbone import (  # noqa: E501
    StableDiffusion1_4Backbone,
)
from keras_hub.src.models.stable_diffusion_1_4.stable_diffusion_1_4_text_to_image_preprocessor import (  # noqa: E501
    StableDiffusion1_4TextToImagePreprocessor,
)
from keras_hub.src.models.text_to_image import TextToImage


@keras_hub_export("keras_hub.models.StableDiffusion1_4TextToImage")
class StableDiffusion1_4TextToImage(TextToImage):
    """An end-to-end Stable Diffusion v1.4 model for text-to-image generation.

    This model has a `generate()` method, which generates images based on a
    prompt.

    Args:
        backbone: A `keras_hub.models.StableDiffusion1_4Backbone` instance.
        preprocessor: A
            `keras_hub.models.StableDiffusion1_4TextToImagePreprocessor`
            instance.

    Examples:

    Use `generate()` to do image generation.
    ```python
    text_to_image = keras_hub.models.StableDiffusion1_4TextToImage.from_preset(
        "stable_diffusion_1_4", image_shape=(512, 512, 3)
    )
    text_to_image.generate(
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )

    # Generate with batched prompts.
    text_to_image.generate(
        ["cute wallpaper art of a cat", "cute wallpaper art of a dog"]
    )

    # Generate with different `num_steps` and `guidance_scale`.
    text_to_image.generate(
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        num_steps=25,
        guidance_scale=7.5,
    )

    # Generate with `negative_prompts`.
    text_to_image.generate(
        {
            "prompts": "Astronaut in a jungle, detailed, 8k",
            "negative_prompts": "green color",
        }
    )
    ```
    """

    backbone_cls = StableDiffusion1_4Backbone
    preprocessor_cls = StableDiffusion1_4TextToImagePreprocessor

    def __init__(self, backbone, preprocessor, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = backbone.input
        outputs = backbone.output
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "Currently, `fit` is not supported for "
            "`StableDiffusion1_4TextToImage`."
        )

    def generate_step(self, latents, token_ids, num_steps, guidance_scale):
        """A compilable generation function for a batch of inputs.

        Args:
            latents: A (batch_size, height, width, channels) tensor containing
                the latents to start generation from. Typically, this tensor is
                sampled from the Gaussian distribution.
            token_ids: A pair of (batch_size, num_tokens) tensors containing the
                tokens for the prompts and negative prompts.
            num_steps: int. The number of diffusion steps to take.
            guidance_scale: float. The classifier-free guidance scale.
        """
        token_ids, negative_token_ids = token_ids

        # Encode prompts.
        embeddings = self.backbone.encode_text_step(
            token_ids, negative_token_ids
        )

        # Denoise. The DPM-Solver++ (2M) step is second-order, so the previous
        # data prediction is carried alongside the latents.
        def body_fun(step, state):
            latents, prev_x0 = state
            return self.backbone.denoise_step(
                latents,
                prev_x0,
                embeddings,
                step,
                num_steps,
                guidance_scale,
            )

        latents, _ = ops.fori_loop(
            0, num_steps, body_fun, (latents, ops.zeros_like(latents))
        )

        # Decode.
        return self.backbone.decode_step(latents)

    def generate(
        self,
        inputs,
        num_steps=25,
        guidance_scale=7.5,
        seed=None,
    ):
        self.backbone.configure_scheduler(num_steps)
        return super().generate(
            inputs,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
