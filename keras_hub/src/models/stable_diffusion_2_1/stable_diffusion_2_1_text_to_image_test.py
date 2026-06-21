from keras import backend
from keras import distribution

from keras_hub.src.models.clip.clip_preprocessor import CLIPPreprocessor
from keras_hub.src.models.clip.clip_text_encoder import CLIPTextEncoder
from keras_hub.src.models.clip.clip_tokenizer import CLIPTokenizer
from keras_hub.src.models.stable_diffusion_2_1.stable_diffusion_2_1_backbone import (  # noqa: E501
    StableDiffusion2_1Backbone,
)
from keras_hub.src.models.stable_diffusion_2_1.stable_diffusion_2_1_text_to_image import (  # noqa: E501
    StableDiffusion2_1TextToImage,
)
from keras_hub.src.models.stable_diffusion_2_1.stable_diffusion_2_1_text_to_image_preprocessor import (  # noqa: E501
    StableDiffusion2_1TextToImagePreprocessor,
)
from keras_hub.src.models.vae.vae_backbone import VAEBackbone
from keras_hub.src.tests.test_case import TestCase


class StableDiffusion2_1TextToImageTest(TestCase):
    def setUp(self):
        # Instantiate the preprocessor.
        merges = ["a i", "p l", "n e</w>", "p o", "r t</w>", "ai r", "pl a"]
        merges += ["po rt</w>", "pla ne</w>"]
        vocab = []
        for merge in merges:
            a, b = merge.split(" ")
            vocab.extend([a, b, a + b])
        vocab += ["<|endoftext|>", "<|startoftext|>"]
        vocab = sorted(set(vocab))  # Remove duplicates
        vocab = dict([(token, i) for i, token in enumerate(vocab)])
        clip_tokenizer = CLIPTokenizer(vocab, merges, pad_with_end_token=True)
        clip_preprocessor = CLIPPreprocessor(clip_tokenizer)
        self.preprocessor = StableDiffusion2_1TextToImagePreprocessor(
            clip_preprocessor
        )

        # TODO: JAX CPU doesn't support float16 in `nn.dot_product_attention`.
        is_jax_cpu = (
            backend.backend() == "jax"
            and "cpu" in distribution.list_devices()[0].lower()
        )
        self.backbone = StableDiffusion2_1Backbone(
            clip=CLIPTextEncoder(
                20,
                64,
                64,
                2,
                2,
                128,
                "gelu",
                dtype="float16" if not is_jax_cpu else None,
                name="clip",
            ),
            vae=VAEBackbone(
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
            ),
            unet_block_out_channels=(32, 64),
            unet_layers_per_block=1,
            unet_head_dim=32,
            image_shape=(64, 64, 3),
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }

    def test_generate(self):
        text_to_image = StableDiffusion2_1TextToImage(**self.init_kwargs)
        seed = 42
        # String input.
        prompt = ["airplane"]
        negative_prompt = [""]
        output = text_to_image.generate(
            {"prompts": prompt, "negative_prompts": negative_prompt},
            seed=seed,
        )
        # Int tensor input.
        prompt_ids = self.preprocessor.generate_preprocess(prompt)
        negative_prompt_ids = self.preprocessor.generate_preprocess(
            negative_prompt
        )
        text_to_image.preprocessor = None
        output2 = text_to_image.generate(
            {"prompts": prompt_ids, "negative_prompts": negative_prompt_ids},
            seed=seed,
        )
        self.assertAllClose(output, output2)

    def test_generate_compilation(self):
        text_to_image = StableDiffusion2_1TextToImage(**self.init_kwargs)
        # Assert we do not recompile with successive calls.
        text_to_image.generate("airplane")
        first_fn = text_to_image.generate_function
        text_to_image.generate("airplane")
        second_fn = text_to_image.generate_function
        self.assertEqual(first_fn, second_fn)

    def test_generate_varying_num_steps(self):
        # Generating at one step count then another on the same instance must
        # match a fresh instance at the second count. Regression test for the
        # cached generate function reusing a scheduler traced for a different
        # `num_steps` (which indexed stale per-step arrays out of range and
        # produced a near-solid-color image).
        reuse = StableDiffusion2_1TextToImage(**self.init_kwargs)
        reuse.generate("airplane", num_steps=3, seed=42)
        reuse_output = reuse.generate("airplane", num_steps=5, seed=42)
        # The function must retrace when the step count changes.
        self.assertEqual(reuse._generate_num_steps, 5)

        fresh = StableDiffusion2_1TextToImage(**self.init_kwargs)
        for a, b in zip(fresh.weights, reuse.weights):
            a.assign(b.numpy())
        fresh_output = fresh.generate("airplane", num_steps=5, seed=42)
        self.assertAllClose(reuse_output, fresh_output)
