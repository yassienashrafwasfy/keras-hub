import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.clip.clip_backbone import CLIPBackbone
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = CLIPBackbone.from_preset(
            "hf://wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
        )
        # Test with dummy image and token inputs.
        images = np.ones((1, 224, 224, 3), dtype="float32")
        token_ids = np.ones((1, 77), dtype="int32")
        outputs = model.predict({"images": images, "token_ids": token_ids})
        # Check output shapes.
        self.assertEqual(outputs["vision_logits"].shape, (1, 1))
        self.assertEqual(outputs["text_logits"].shape, (1, 1))

    @pytest.mark.extra_large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
            load_weights=False,
        )
        self.assertIsInstance(model, CLIPBackbone)
