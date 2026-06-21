"""Convert Deep Compression Autoencoder (DC-AE) checkpoints.

python tools/checkpoint_conversion/convert_dc_ae_checkpoints.py \
    --preset dc_ae_f32c32_in_1.0 \
    --upload_uri kaggle://kerashub/dcae/keras/dc_ae_f32c32_in_1.0
"""

import os

# Use the torch backend for conversion. Must be set before `keras` is imported.
os.environ["KERAS_BACKEND"] = "torch"

import gc  # noqa: E402

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from diffusers import AutoencoderDC  # noqa: E402

import keras_hub  # noqa: E402
from keras_hub.src.models.dc_ae.dc_ae_backbone import DCAEBackbone  # noqa: E402
from keras_hub.src.models.dc_ae.dc_ae_layers import DCDownBlock2d  # noqa: E402
from keras_hub.src.models.dc_ae.dc_ae_layers import DCResBlock  # noqa: E402
from keras_hub.src.models.dc_ae.dc_ae_layers import DCUpBlock2d  # noqa: E402
from keras_hub.src.models.dc_ae.dc_ae_layers import (  # noqa: E402
    EfficientViTBlock,
)
from keras_hub.src.models.dc_ae.dc_ae_layers import RMSNorm2D  # noqa: E402
from keras_hub.src.utils.transformers.safetensor_utils import (  # noqa: E402
    SafetensorLoader,
)

FLAGS = flags.FLAGS

PRESET_MAP = {
    "dc_ae_f32c32_in_1.0": {
        "root": "hf://mit-han-lab/dc-ae-f32c32-in-1.0-diffusers",
        "weights": "diffusion_pytorch_model.safetensors",
        "scale": 0.3189,
        # 32x compression; a 512x512 image becomes a 16x16x32 latent.
        "image_size": 512,
    },
}

flags.DEFINE_string(
    "preset",
    "dc_ae_f32c32_in_1.0",
    f"Must be one of {','.join(PRESET_MAP.keys())}",
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://kerashub/dcae/keras/{preset}"',
    required=False,
)


def convert_model(preset):
    config = PRESET_MAP[preset]
    image_size = config["image_size"]
    # The defaults of `DCAEBackbone` already match `dc-ae-f32c32-in-1.0`.
    return DCAEBackbone(
        scale=config["scale"],
        image_shape=(image_size, image_size, 3),
        dtype="float32",
        name="dc_ae_backbone",
    )


def convert_weights(preset, keras_model):
    config = PRESET_MAP[preset]

    def port_conv2d(loader, conv, key):
        loader.port_weight(
            conv.kernel,
            f"{key}.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )
        if conv.bias is not None:
            loader.port_weight(conv.bias, f"{key}.bias")

    def port_depthwise(loader, conv, key):
        # torch depthwise weight `(C, 1, kh, kw)` -> keras `(kh, kw, C, 1)`.
        loader.port_weight(
            conv.kernel,
            f"{key}.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 0, 1)),
        )
        if conv.bias is not None:
            loader.port_weight(conv.bias, f"{key}.bias")

    def port_dense(loader, dense, key):
        loader.port_weight(
            dense.kernel, f"{key}.weight", hook_fn=lambda x, _: x.T
        )
        if dense.bias is not None:
            loader.port_weight(dense.bias, f"{key}.bias")

    def port_norm(loader, norm, key):
        if isinstance(norm, RMSNorm2D):
            loader.port_weight(norm.gamma, f"{key}.weight")
            loader.port_weight(norm.beta, f"{key}.bias")
        else:  # keras.layers.BatchNormalization
            loader.port_weight(norm.gamma, f"{key}.weight")
            loader.port_weight(norm.beta, f"{key}.bias")
            loader.port_weight(norm.moving_mean, f"{key}.running_mean")
            loader.port_weight(norm.moving_variance, f"{key}.running_var")

    def port_attention(loader, attn, key):
        port_dense(loader, attn.to_q, f"{key}.to_q")
        port_dense(loader, attn.to_k, f"{key}.to_k")
        port_dense(loader, attn.to_v, f"{key}.to_v")
        for j, projection in enumerate(attn.multiscale_projections):
            port_depthwise(
                loader,
                projection.proj_in,
                f"{key}.to_qkv_multiscale.{j}.proj_in",
            )
            port_conv2d(
                loader,
                projection.proj_out,
                f"{key}.to_qkv_multiscale.{j}.proj_out",
            )
        port_dense(loader, attn.to_out, f"{key}.to_out")
        port_norm(loader, attn.norm_out, f"{key}.norm_out")

    def port_glumbconv(loader, glumb, key):
        port_conv2d(loader, glumb.conv_inverted, f"{key}.conv_inverted")
        port_depthwise(loader, glumb.conv_depth, f"{key}.conv_depth")
        port_conv2d(loader, glumb.conv_point, f"{key}.conv_point")
        if glumb.norm is not None:
            port_norm(loader, glumb.norm, f"{key}.norm")

    def port_block(loader, block, key):
        if isinstance(block, DCResBlock):
            port_conv2d(loader, block.conv1, f"{key}.conv1")
            port_conv2d(loader, block.conv2, f"{key}.conv2")
            port_norm(loader, block.norm, f"{key}.norm")
        elif isinstance(block, EfficientViTBlock):
            port_attention(loader, block.attn, f"{key}.attn")
            port_glumbconv(loader, block.conv_out, f"{key}.conv_out")
        elif isinstance(block, (DCDownBlock2d, DCUpBlock2d)):
            port_conv2d(loader, block.conv, f"{key}.conv")
        else:
            raise ValueError(f"Unhandled block {type(block)} at {key}")

    root = config["root"]
    with SafetensorLoader(root, prefix="", fname=config["weights"]) as loader:
        # === Encoder ===
        encoder = keras_model.encoder
        if isinstance(encoder.conv_in, DCDownBlock2d):
            port_conv2d(loader, encoder.conv_in.conv, "encoder.conv_in.conv")
        else:
            port_conv2d(loader, encoder.conv_in, "encoder.conv_in")
        for block, key in zip(encoder.blocks, encoder.block_keys):
            port_block(loader, block, key)
        port_conv2d(loader, encoder.conv_out, "encoder.conv_out")

        # === Decoder ===
        decoder = keras_model.decoder
        port_conv2d(loader, decoder.conv_in, "decoder.conv_in")
        for block, key in zip(decoder.blocks, decoder.block_keys):
            port_block(loader, block, key)
        port_norm(loader, decoder.norm_out, "decoder.norm_out")
        if isinstance(decoder.conv_out, DCUpBlock2d):
            port_conv2d(loader, decoder.conv_out.conv, "decoder.conv_out.conv")
        else:
            port_conv2d(loader, decoder.conv_out, "decoder.conv_out")


def validate_output(preset, keras_model):
    torch.set_grad_enabled(False)
    config = PRESET_MAP[preset]
    image_size = config["image_size"]
    hf_repo_id = config["root"].replace("hf://", "", 1)

    rng = np.random.default_rng(0)
    images = rng.standard_normal(
        (1, image_size, image_size, 3), dtype=np.float32
    )

    # === keras (channels_last). ===
    keras_images = keras.ops.convert_to_tensor(images)
    keras_latent = keras.ops.convert_to_numpy(keras_model.encode(keras_images))
    keras_recon = keras.ops.convert_to_numpy(
        keras_model.decode(keras_model.encode(keras_images))
    )

    # === diffusers (channels_first). ===
    hf_model = AutoencoderDC.from_pretrained(
        hf_repo_id, torch_dtype=torch.float32
    )
    hf_model.eval()
    hf_images = torch.from_numpy(np.transpose(images, (0, 3, 1, 2)))
    hf_latent = hf_model.encode(hf_images).latent
    hf_recon = hf_model.decode(hf_latent).sample.numpy()
    hf_latent = hf_latent.numpy()
    del hf_model
    gc.collect()

    # Compare latents (move HF to channels_last).
    hf_latent_cl = np.transpose(hf_latent, (0, 2, 3, 1))
    hf_recon_cl = np.transpose(hf_recon, (0, 2, 3, 1))
    latent_diff = np.mean(np.abs(keras_latent - hf_latent_cl))
    recon_diff = np.mean(np.abs(keras_recon - hf_recon_cl))
    print("🔶 Latent shape (keras):", keras_latent.shape)
    print("🔶 Latent mean abs diff:", latent_diff)
    print("🔶 Reconstruction mean abs diff:", recon_diff)
    assert latent_diff < 1e-3, f"Latent mismatch: {latent_diff}"
    assert recon_diff < 1e-3, f"Reconstruction mismatch: {recon_diff}"
    print("🔶 Validation passed.")


def main(_):
    preset = FLAGS.preset
    print(f"🏃 Converting {preset}")
    keras_model = convert_model(preset)
    print("✅ KerasHub model loaded.")
    convert_weights(preset, keras_model)
    print("✅ Weights converted.")
    validate_output(preset, keras_model)
    print("✅ Output validated.")

    keras_model.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}.")


if __name__ == "__main__":
    app.run(main)
