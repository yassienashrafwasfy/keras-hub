import numpy as np

from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.tests.test_case import TestCase


class GemmaLoraTest(TestCase):
    def setUp(self):
        self._init_kwargs = {
            "vocabulary_size": 50,
            "num_layers": 2,
            "num_query_heads": 2,
            "num_key_value_heads": 2,
            "hidden_dim": 32,
            "intermediate_dim": 16,
            "head_dim": 16,
            "layer_norm_epsilon": 1e-6,
        }

    def test_lora(self):
        self.run_lora_test(
            cls=GemmaBackbone,
            init_kwargs=self._init_kwargs,
            input_data={
                "token_ids": np.ones((2, 5), dtype="int32"),
                "padding_mask": np.ones((2, 5), dtype="int32"),
            },
            expected_num_trainable_weights=4 * 2,
            expected_num_non_trainable_weights=20,
        )

    def test_lora_target_layer_names(self):
        self.run_lora_test(
            cls=GemmaBackbone,
            init_kwargs=self._init_kwargs,
            input_data={
                "token_ids": np.ones((2, 5), dtype="int32"),
                "padding_mask": np.ones((2, 5), dtype="int32"),
            },
            target_layer_names=["query"],
            expected_num_trainable_weights=2 * 2,
            expected_num_non_trainable_weights=20,
        )
