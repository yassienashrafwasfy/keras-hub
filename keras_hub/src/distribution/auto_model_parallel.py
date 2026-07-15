import math

import keras

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.distribution.AutoModelParallel")
class AutoModelParallel(keras.distribution.ModelParallel):
    """Automatic tensor-parallel distribution for any Keras model.

    This distribution shards model weights across all available accelerators
    without requiring a per-model `LayoutMap`. Instead of matching variable
    paths against hand-written regexes, the sharding for each variable is
    computed from its shape when the variable is created:

    - Variables with rank < 2 (biases, norm scales) are replicated.
    - Variables smaller than `min_size_to_shard` elements are replicated.
    - Otherwise the variable is sharded along its largest dimension that is
      divisible by the number of devices, with ties going to the trailing
      axis (the output features dimension of dense/einsum kernels).

    Because sharding is derived from shapes, this works for any `Backbone`
    (transformers, convnets, MoE) with no per-architecture code, and cannot
    silently break when a layer is renamed. An explicit `LayoutMap` can still
    be passed to override the automatic choice for specific variables; any
    variable it matches bypasses the automatic rule.

    This distribution is only supported on the JAX backend, where the XLA
    GSPMD partitioner inserts the necessary collective communication
    automatically.

    Example:
    ```python
    # Shard a backbone across all available devices.
    distribution = keras_hub.distribution.AutoModelParallel()
    with distribution.scope():
        backbone = keras_hub.models.Backbone.from_preset("gemma_2b_en")

    # Or equivalently, via `from_preset`:
    backbone = keras_hub.models.Backbone.from_preset(
        "gemma_2b_en", sharding="auto"
    )
    ```

    Args:
        devices: Optional list of devices to shard over. Defaults to all
            devices from `keras.distribution.list_devices()`.
        layout_map: Optional `keras.distribution.LayoutMap` with explicit
            overrides. Variables matched by this map are laid out as
            specified instead of automatically. If provided, its device mesh
            is used and `devices` is ignored.
        batch_dim_name: The mesh axis name for the data dimension. Defaults
            to `"batch"`.
        model_dim_name: The mesh axis name for the model dimension. Defaults
            to `"model"`.
        min_size_to_shard: Minimum number of elements a variable must have
            to be sharded. Smaller variables are replicated. Defaults to
            `2**16`.
        **kwargs: Additional kwargs passed to
            `keras.distribution.ModelParallel`.
    """

    def __init__(
        self,
        devices=None,
        layout_map=None,
        batch_dim_name="batch",
        model_dim_name="model",
        min_size_to_shard=2**16,
        **kwargs,
    ):
        if keras.config.backend() != "jax":
            raise ValueError(
                "`AutoModelParallel` is only supported on the JAX backend. "
                "`keras.distribution` has no implementation for the "
                f"`{keras.config.backend()}` backend. Set the environment "
                "variable `KERAS_BACKEND=jax` before importing Keras."
            )
        if layout_map is not None:
            device_mesh = layout_map.device_mesh
        else:
            if devices is None:
                devices = keras.distribution.list_devices()
            device_mesh = keras.distribution.DeviceMesh(
                shape=(1, len(devices)),
                axis_names=(batch_dim_name, model_dim_name),
                devices=devices,
            )
            layout_map = keras.distribution.LayoutMap(device_mesh)
        if model_dim_name not in device_mesh.axis_names:
            raise ValueError(
                f"`model_dim_name={model_dim_name}` is not found in the "
                f"device mesh axis names: {device_mesh.axis_names}"
            )
        super().__init__(
            layout_map=layout_map, batch_dim_name=batch_dim_name, **kwargs
        )
        self.model_dim_name = model_dim_name
        self.min_size_to_shard = min_size_to_shard
        axis_sizes = dict(zip(device_mesh.axis_names, device_mesh.shape))
        self._num_model_shards = axis_sizes[model_dim_name]

    def get_variable_layout(self, variable):
        # A layout already assigned to the variable always wins.
        if getattr(variable, "_layout", None) is not None:
            return variable._layout
        # Explicit user overrides from the layout map come next.
        explicit_layout = self._layout_map[variable.path]
        if explicit_layout is not None:
            return explicit_layout
        return keras.distribution.TensorLayout(
            self._auto_shard_spec(variable.shape), self.device_mesh
        )

    def _auto_shard_spec(self, shape):
        """Compute a sharding spec for a variable shape.

        Returns a list with `self.model_dim_name` on the axis to shard, or
        all `None` (replicated) when no axis is shardable.
        """
        spec = [None] * len(shape)
        if self._num_model_shards <= 1 or len(shape) < 2:
            return spec
        if math.prod(shape) < self.min_size_to_shard:
            return spec
        candidates = [
            axis
            for axis, dim in enumerate(shape)
            if dim >= self._num_model_shards
            and dim % self._num_model_shards == 0
        ]
        if not candidates:
            return spec
        best = max(candidates, key=lambda axis: (shape[axis], axis))
        spec[best] = self.model_dim_name
        return spec
