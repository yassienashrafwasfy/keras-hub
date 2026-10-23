import numpy as np

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.tests.test_case import TestCase


class Gemma3LoraTest(TestCase):
    def setUp(self):
        self.text_init_kwargs = {
            # vocabulary
            "vocabulary_size": 256,
            # image
            "image_size": 16,
            # model
            "num_layers": 6,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "head_dim": 4,
            # other model args
            "query_head_dim_normalize": True,
            "use_query_key_norm": True,
            "use_post_ffw_norm": True,
            "use_post_attention_norm": True,
            "final_logit_soft_cap": None,
            "attention_logit_soft_cap": None,
            "use_sliding_window_attention": True,
            "sliding_window_size": 1024,
            "vision_encoder": None,
        }

    def test_text_lora(self):
        self.run_lora_test(
            cls=Gemma3Backbone,
            init_kwargs=self.text_init_kwargs,
            input_data={
                "token_ids": np.ones((2, 5), dtype="int32"),
                "padding_mask": np.ones((2, 5), dtype="int32"),
            },
            expected_num_trainable_weights=24,
            expected_num_non_trainable_weights=80,
        )
