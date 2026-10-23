import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_backbone import (
    Qwen3_5MoeBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class TestQwen3_5MoeConverter(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        self.run_hf_class_test(
            "hf://yujiepan/qwen3.5-moe-tiny-random",
            {Qwen3_5MoeBackbone: Qwen3_5MoeBackbone},
        )

    @pytest.mark.large
    def test_class_detection(self):
        self.run_hf_class_test(
            "hf://yujiepan/qwen3.5-moe-tiny-random",
            {Backbone: Qwen3_5MoeBackbone},
        )
