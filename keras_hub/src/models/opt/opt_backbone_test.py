import pytest
from keras import ops

from keras_hub.src.models.opt.opt_backbone import OPTBackbone
from keras_hub.src.tests.test_case import BackboneTestMixin
from keras_hub.src.tests.test_case import TestCase


class OPTBackboneTest(BackboneTestMixin, TestCase):
    backbone_cls = OPTBackbone
    expected_output_shape = (2, 5, 2)

    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 2,
            "intermediate_dim": 4,
            "max_sequence_length": 5,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=OPTBackbone,
            preset="opt_125m_en",
            input_data={
                "token_ids": ops.array([[133, 2119, 6219, 23602, 4]]),
                "padding_mask": ops.ones((1, 5), dtype="int32"),
            },
            expected_output_shape=(1, 5, 768),
            # The forward pass from a preset should be stable!
            expected_partial_output=ops.array(
                [-0.246, -1.004, -0.072, 0.097, 0.533]
            ),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in OPTBackbone.presets:
            self.run_preset_test(
                cls=OPTBackbone,
                preset=preset,
                input_data=self.input_data,
            )
