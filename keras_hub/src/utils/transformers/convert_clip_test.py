import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.clip.clip_backbone import CLIPBackbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.convert_clip import (
    convert_backbone_config,
)


class TestTask(TestCase):
    def test_backbone_config_defaults(self):
        # Long-CLIP style config: omits keys that match the HuggingFace
        # defaults (`num_hidden_layers`, `vocab_size`, `image_size`) and uses a
        # stretched 248-token text context.
        transformers_config = {
            "projection_dim": 768,
            "text_config": {
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "max_position_embeddings": 248,
            },
            "vision_config": {
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_attention_heads": 16,
                "num_hidden_layers": 24,
                "patch_size": 14,
            },
        }
        config = convert_backbone_config(transformers_config)
        text_encoder = config["text_encoder"]
        vision_encoder = config["vision_encoder"]
        self.assertEqual(config["projection_dim"], 768)
        # Defaults filled in from `.get(...)`.
        self.assertEqual(text_encoder.vocabulary_size, 49408)
        self.assertEqual(text_encoder.num_layers, 12)
        self.assertEqual(text_encoder.max_sequence_length, 248)
        self.assertEqual(vision_encoder.image_shape, (224, 224, 3))
        self.assertEqual(vision_encoder.num_layers, 24)
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
