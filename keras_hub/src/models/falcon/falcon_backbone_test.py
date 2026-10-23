from keras import ops

from keras_hub.src.models.falcon.falcon_backbone import FalconBackbone
from keras_hub.src.tests.test_case import BackboneTestMixin
from keras_hub.src.tests.test_case import TestCase


class FalconBackboneTest(BackboneTestMixin, TestCase):
    backbone_cls = FalconBackbone
    expected_output_shape = (2, 5, 16)

    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_attention_heads": 8,
            "hidden_dim": 16,
            "intermediate_dim": 32,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }
