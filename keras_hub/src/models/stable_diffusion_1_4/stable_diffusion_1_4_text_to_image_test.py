from keras import backend
from keras import distribution

from keras_hub.src.models.clip.clip_preprocessor import CLIPPreprocessor
from keras_hub.src.models.clip.clip_text_encoder import CLIPTextEncoder
from keras_hub.src.models.clip.clip_tokenizer import CLIPTokenizer
from keras_hub.src.models.stable_diffusion_1_4.stable_diffusion_1_4_backbone import (  # noqa: E501
    StableDiffusion1_4Backbone,
)
from keras_hub.src.models.stable_diffusion_1_4.stable_diffusion_1_4_text_to_image import (  # noqa: E501
    StableDiffusion1_4TextToImage,
)
from keras_hub.src.models.stable_diffusion_1_4.stable_diffusion_1_4_text_to_image_preprocessor import (  # noqa: E501
    StableDiffusion1_4TextToImagePreprocessor,
)
from keras_hub.src.models.vae.vae_backbone import VAEBackbone
from keras_hub.src.tests.test_case import TestCase


class StableDiffusion1_4TextToImageTest(TestCase):
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
        self.preprocessor = StableDiffusion1_4TextToImagePreprocessor(
            clip_preprocessor
        )

        # TODO: JAX CPU doesn't support float16 in `nn.dot_product_attention`.
        is_jax_cpu = (
            backend.backend() == "jax"
            and "cpu" in distribution.list_devices()[0].lower()
        )
        self.backbone = StableDiffusion1_4Backbone(
            clip=CLIPTextEncoder(
                20,
                64,
                64,
                2,
                2,
                128,
                "quick_gelu",
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
            unet_num_heads=8,
            image_shape=(64, 64, 3),
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
        }

    def test_generate(self):
        text_to_image = StableDiffusion1_4TextToImage(**self.init_kwargs)
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
        text_to_image = StableDiffusion1_4TextToImage(**self.init_kwargs)
        # Assert we do not recompile with successive calls.
        text_to_image.generate("airplane")
        first_fn = text_to_image.generate_function
        text_to_image.generate("airplane")
        second_fn = text_to_image.generate_function
        self.assertEqual(first_fn, second_fn)
