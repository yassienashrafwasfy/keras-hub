import keras
import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.vit.vit_backbone import ViTBackbone
from keras_hub.src.models.vit.vit_image_classifier import ViTImageClassifier
from keras_hub.src.tests.test_case import TestCase


@pytest.mark.skipif(
    keras.backend.backend() == "tensorflow",
    reason="TensorFlow GPU CI OOM (ResourceExhaustedError)",
)
class TestViTConverter(TestCase):
    @pytest.mark.extra_large
    def test_convert_preset(self):
        model = ViTImageClassifier.from_preset(
            "hf://google/vit-base-patch16-224"
        )
        output = model.predict(
            {"images": np.ones((1, 224, 224, 3), dtype="float32")}
        )
        self.assertEqual(output.shape, (1, 1000))

    @pytest.mark.extra_large
    def test_class_detection(self):
        self.run_hf_class_test(
            "hf://google/vit-base-patch16-224",
            {ImageClassifier: ViTImageClassifier, Backbone: ViTBackbone},
        )
