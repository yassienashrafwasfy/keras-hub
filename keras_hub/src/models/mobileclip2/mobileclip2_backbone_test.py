import pytest
from keras import ops

from keras_hub.src.models.mobileclip2.mobileclip2_backbone import (
    MobileCLIP2Backbone,
)
from keras_hub.src.models.mobileclip2.mobileclip2_text_encoder import (
    MobileCLIP2TextEncoder,
)
from keras_hub.src.models.mobileclip2.mobileclip2_vision_encoder import (
    MobileCLIP2VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class MobileCLIP2BackboneTest(TestCase):
    def setUp(self):
        vision_encoder = MobileCLIP2VisionEncoder(
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            intermediate_dim=128,
            stem_channels=(16, 16, 64),
            name="vision_encoder",
        )
        text_encoder = MobileCLIP2TextEncoder(
            vocabulary_size=64,
            embedding_dim=64,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            intermediate_dim=128,
            name="text_encoder",
        )
        self.init_kwargs = {
            "vision_encoder": vision_encoder,
            "text_encoder": text_encoder,
            "projection_dim": 64,
        }
        self.input_data = {
            "images": ops.ones((2, 224, 224, 3)),
            "token_ids": ops.ones((2, 77), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=MobileCLIP2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "vision_logits": (2, 2),
                "text_logits": (2, 2),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileCLIP2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
