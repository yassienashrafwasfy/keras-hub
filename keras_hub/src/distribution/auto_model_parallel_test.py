import keras
import numpy as np
from keras import ops

from keras_hub.src.distribution.auto_model_parallel import AutoModelParallel
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.bert.bert_backbone import BertBackbone
from keras_hub.src.tests.test_case import TestCase


class AutoModelParallelTest(TestCase):
    def setUp(self):
        # Dims are multiples of 4 so an axis is shardable on a 4-device mesh.
        self.init_kwargs = {
            "vocabulary_size": 64,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "max_sequence_length": 8,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 8), dtype="int32"),
            "segment_ids": ops.zeros((2, 8), dtype="int32"),
            "padding_mask": ops.ones((2, 8), dtype="int32"),
        }

    def skip_if_not_multi_device_jax(self):
        if keras.backend.backend() != "jax":
            self.skipTest("`AutoModelParallel` requires the JAX backend.")
        if len(keras.distribution.list_devices()) < 2:
            self.skipTest(
                "`AutoModelParallel` testing requires multiple devices. Run "
                "with XLA_FLAGS=--xla_force_host_platform_device_count=4."
            )

    def test_non_jax_backend_raises(self):
        if keras.backend.backend() == "jax":
            self.skipTest("This test covers non-JAX backends.")
        with self.assertRaises(ValueError):
            AutoModelParallel()

    def test_auto_shard_spec(self):
        self.skip_if_not_multi_device_jax()
        distribution = AutoModelParallel(min_size_to_shard=0)
        shards = distribution._num_model_shards
        self.assertGreater(shards, 1)
        # Rank < 2 is always replicated.
        self.assertEqual(distribution._auto_shard_spec(()), [])
        self.assertEqual(distribution._auto_shard_spec((shards,)), [None])
        # Largest divisible axis is sharded.
        self.assertEqual(
            distribution._auto_shard_spec((8 * shards, 4 * shards)),
            ["model", None],
        )
        # Ties go to the trailing axis.
        self.assertEqual(
            distribution._auto_shard_spec((4 * shards, 4 * shards)),
            [None, "model"],
        )
        # Non-divisible dims are skipped in favor of divisible ones.
        self.assertEqual(
            distribution._auto_shard_spec((8 * shards + 1, 4 * shards)),
            [None, "model"],
        )
        # Fully non-divisible shapes are replicated.
        self.assertEqual(
            distribution._auto_shard_spec((shards + 1, shards + 1)),
            [None, None],
        )
        # Shapes below the size threshold are replicated.
        distribution = AutoModelParallel(min_size_to_shard=2**30)
        self.assertEqual(
            distribution._auto_shard_spec((8 * shards, 4 * shards)),
            [None, None],
        )

    def test_backbone_weights_sharded(self):
        self.skip_if_not_multi_device_jax()
        distribution = AutoModelParallel(min_size_to_shard=0)
        shards = distribution._num_model_shards
        with distribution.scope():
            model = BertBackbone(**self.init_kwargs)

        sharded_paths = []
        for w in model.weights:
            spec = tuple(w.value.sharding.spec)
            if "model" in spec:
                sharded_paths.append(w.path)
                axis = spec.index("model")
                # The sharded axis must be divisible by the shard count.
                self.assertEqual(w.shape[axis] % shards, 0)
            if len(w.shape) < 2:
                # Biases and norm scales must be replicated.
                self.assertNotIn("model", spec)
        # Every kernel/embedding of this size should have been sharded.
        self.assertTrue(
            any("token_embedding/embeddings" in path for path in sharded_paths)
        )
        num_kernels = len([w for w in model.weights if len(w.shape) >= 2])
        self.assertEqual(len(sharded_paths), num_kernels)

    def test_layout_map_override_wins(self):
        self.skip_if_not_multi_device_jax()
        devices = keras.distribution.list_devices()
        device_mesh = keras.distribution.DeviceMesh(
            shape=(1, len(devices)),
            axis_names=("batch", "model"),
            devices=devices,
        )
        layout_map = keras.distribution.LayoutMap(device_mesh)
        # Force the token embedding to stay replicated.
        layout_map["token_embedding/embeddings"] = (None, None)
        distribution = AutoModelParallel(
            layout_map=layout_map, min_size_to_shard=0
        )
        with distribution.scope():
            model = BertBackbone(**self.init_kwargs)
        for w in model.weights:
            if "token_embedding/embeddings" in w.path:
                self.assertNotIn("model", tuple(w.value.sharding.spec))

    def test_sharded_numerics_match_replicated(self):
        self.skip_if_not_multi_device_jax()
        distribution = AutoModelParallel(min_size_to_shard=0)
        with distribution.scope():
            sharded = BertBackbone(**self.init_kwargs)
        replicated = BertBackbone(**self.init_kwargs)
        replicated.set_weights(sharded.get_weights())

        sharded_output = sharded(self.input_data)
        replicated_output = replicated(self.input_data)
        self.assertAllClose(
            np.asarray(sharded_output["sequence_output"]),
            np.asarray(replicated_output["sequence_output"]),
            atol=1e-5,
        )

    def test_from_preset_sharding(self):
        self.skip_if_not_multi_device_jax()
        preset_dir = self.get_temp_dir()
        model = BertBackbone(**self.init_kwargs)
        model.save_to_preset(preset_dir)

        reference_output = model(self.input_data)
        distribution = AutoModelParallel(min_size_to_shard=0)
        restored = Backbone.from_preset(preset_dir, sharding=distribution)

        self.assertTrue(
            any(
                "model" in tuple(w.value.sharding.spec)
                for w in restored.weights
            )
        )
        restored_output = restored(self.input_data)
        self.assertAllClose(
            np.asarray(reference_output["sequence_output"]),
            np.asarray(restored_output["sequence_output"]),
            atol=1e-5,
        )

    def test_from_preset_invalid_sharding(self):
        preset_dir = self.get_temp_dir()
        model = BertBackbone(**self.init_kwargs)
        model.save_to_preset(preset_dir)
        with self.assertRaises(ValueError):
            Backbone.from_preset(preset_dir, sharding="invalid")
