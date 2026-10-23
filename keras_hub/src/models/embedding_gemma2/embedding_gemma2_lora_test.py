import numpy as np

from keras_hub.src.models.embedding_gemma2.embedding_gemma2_backbone import (
    EmbeddingGemma2Backbone,
)
from keras_hub.src.tests.test_case import TestCase


class EmbeddingGemma2LoraTest(TestCase):
    def setUp(self):
        self.text_init_kwargs = {
            "vocabulary_size": 256,
            "image_size": 16,
            "num_layers": 6,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "head_dim": 4,
            "embedding_dim": 8,
            "use_sliding_window_attention": True,
            "sliding_window_size": 16,
            "vision_encoder": None,
        }

    def test_text_lora(self):
        self.run_lora_test(
            cls=EmbeddingGemma2Backbone,
            init_kwargs=self.text_init_kwargs,
            input_data={
                "token_ids": np.ones((2, 5), dtype="int32"),
                "padding_mask": np.ones((2, 5), dtype="int32"),
                "position_ids": np.tile(
                    np.arange(5, dtype="int32")[None, :], (2, 1)
                ),
            },
        )
