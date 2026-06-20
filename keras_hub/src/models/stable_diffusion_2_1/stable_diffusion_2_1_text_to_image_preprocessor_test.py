from keras_hub.src.models.clip.clip_preprocessor import CLIPPreprocessor
from keras_hub.src.models.clip.clip_tokenizer import CLIPTokenizer
from keras_hub.src.models.stable_diffusion_2_1.stable_diffusion_2_1_text_to_image_preprocessor import (  # noqa: E501
    StableDiffusion2_1TextToImagePreprocessor,
)
from keras_hub.src.tests.test_case import TestCase


class StableDiffusion2_1TextToImagePreprocessorTest(TestCase):
    def setUp(self):
        merges = ["a i", "p l", "n e</w>", "p o", "r t</w>", "ai r", "pl a"]
        merges += ["po rt</w>", "pla ne</w>"]
        vocab = []
        for merge in merges:
            a, b = merge.split(" ")
            vocab.extend([a, b, a + b])
        vocab = sorted(set(vocab))  # Remove duplicates
        vocab += ["<|endoftext|>", "<|startoftext|>"]
        vocab = dict([(token, i) for i, token in enumerate(vocab)])
        clip_tokenizer = CLIPTokenizer(
            vocabulary=vocab, merges=merges, pad_with_end_token=True
        )
        clip_preprocessor = CLIPPreprocessor(clip_tokenizer, sequence_length=8)
        self.init_kwargs = {"clip_preprocessor": clip_preprocessor}
        self.input_data = ["airplane"]

    def test_generate_preprocess(self):
        preprocessor = StableDiffusion2_1TextToImagePreprocessor(
            **self.init_kwargs
        )
        x = preprocessor.generate_preprocess(self.input_data)
        self.assertEqual(tuple(x.shape), (1, 8))
        self.assertAllEqual(x[0], [19, 2, 12, 18, 18, 18, 18, 18])
