import pytest
from keras import ops

from keras_hub.src.models.dc_ae.dc_ae_backbone import DCAEBackbone
from keras_hub.src.tests.test_case import TestCase


class DCAEBackboneTest(TestCase):
    def setUp(self):
        self.height, self.width = 64, 64
        # A tiny 6-stage config that keeps the 32x compression and exercises
        # both `ResBlock` and `EfficientViTBlock` stages and the residual
        # down/upsampling.
        self.init_kwargs = {
            "in_channels": 3,
            "latent_channels": 8,
            "attention_head_dim": 8,
            "encoder_block_out_channels": (8, 16, 32, 32, 64, 64),
            "decoder_block_out_channels": (8, 16, 32, 32, 64, 64),
            "encoder_layers_per_block": (0, 1, 1, 1, 1, 1),
            "decoder_layers_per_block": (0, 1, 1, 1, 1, 1),
            "image_shape": (self.height, self.width, 3),
        }
        self.input_data = ops.ones((2, self.height, self.width, 3))

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=DCAEBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, self.height, self.width, 3),
        )

    def test_encode_decode(self):
        backbone = DCAEBackbone(**self.init_kwargs)
        latents = backbone.encode(self.input_data)
        # 64 // 32 = 2 spatial, 8 latent channels.
        self.assertEqual(tuple(latents.shape), (2, 2, 2, 8))
        reconstructed = backbone.decode(latents)
        self.assertEqual(
            tuple(reconstructed.shape), (2, self.height, self.width, 3)
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DCAEBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
