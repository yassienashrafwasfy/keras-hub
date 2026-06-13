"""Convert Stable Diffusion v1.4 checkpoints.

python tools/checkpoint_conversion/convert_stable_diffusion_1_4_checkpoints.py \
    --preset stable_diffusion_1_4 \
    --upload_uri kaggle://kerashub/stablediffusion1.4/keras/stable_diffusion_1_4
"""

import os
import shutil

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from diffusers import StableDiffusionPipeline
from PIL import Image

import keras_hub
from keras_hub.src.models.clip.clip_preprocessor import CLIPPreprocessor
from keras_hub.src.models.clip.clip_text_encoder import CLIPTextEncoder
from keras_hub.src.models.clip.clip_tokenizer import CLIPTokenizer
from keras_hub.src.models.stable_diffusion_1_4.stable_diffusion_1_4_backbone import (  # noqa: E501
    StableDiffusion1_4Backbone,
)
from keras_hub.src.models.stable_diffusion_1_4.stable_diffusion_1_4_text_to_image import (  # noqa: E501
    StableDiffusion1_4TextToImage,
)
from keras_hub.src.models.stable_diffusion_1_4.stable_diffusion_1_4_text_to_image_preprocessor import (  # noqa: E501
    StableDiffusion1_4TextToImagePreprocessor,
)
from keras_hub.src.models.vae.vae_backbone import VAEBackbone
from keras_hub.src.utils.preset_utils import load_json
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader

FLAGS = flags.FLAGS

PRESET_MAP = {
    "stable_diffusion_1_4": {
        "root": "hf://CompVis/stable-diffusion-v1-4",
        "clip": "text_encoder/model.safetensors",
        "vae": "vae/diffusion_pytorch_model.safetensors",
        "unet": "unet/diffusion_pytorch_model.safetensors",
        "clip_tokenizer": "hf://openai/clip-vit-large-patch14",
        "dtype": "float32",
    },
}

flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
    required=True,
)
flags.DEFINE_string(
    "output_dir",
    "output_dir",
    "The generated image will be saved here.",
    required=False,
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://keras/{variant}/keras/{preset}"',
    required=False,
)


def convert_model(preset, height, width):
    config = PRESET_MAP[preset]
    dtype = config["dtype"]
    vae = VAEBackbone(
        [128, 256, 512, 512],
        [2, 2, 2, 2],
        [512, 512, 256, 128],
        [3, 3, 3, 3],
        sample_channels=8,
        scale=0.18215,
        shift=0.0,
        dtype=dtype,
        name="vae",
    )
    clip = CLIPTextEncoder(
        49408,
        768,
        768,
        12,
        12,
        3072,
        "quick_gelu",
        dtype=dtype,
        name="clip",
    )
    backbone = StableDiffusion1_4Backbone(
        clip=clip,
        vae=vae,
        unet_block_out_channels=(320, 640, 1280, 1280),
        unet_layers_per_block=2,
        unet_num_heads=8,
        image_shape=(height, width, 3),
        dtype=dtype,
        name="stable_diffusion_1_4_backbone",
    )
    return backbone


def convert_preprocessor(preset):
    tokenizer_content = load_json(
        PRESET_MAP[preset]["clip_tokenizer"], "tokenizer.json"
    )
    vocabulary = tokenizer_content["model"]["vocab"]
    merges = tokenizer_content["model"]["merges"]
    clip_tokenizer = CLIPTokenizer(
        vocabulary,
        merges,
        pad_with_end_token=True,
        config_file="clip_tokenizer.json",
        name="clip_tokenizer",
    )
    clip_preprocessor = CLIPPreprocessor(
        clip_tokenizer,
        config_file="clip_preprocessor.json",
        name="clip_preprocessor",
    )
    return StableDiffusion1_4TextToImagePreprocessor(
        clip_preprocessor,
        name="stable_diffusion_1_4_text_to_image_preprocessor",
    )


def convert_weights(preset, keras_model):
    config = PRESET_MAP[preset]

    # === Shared port helpers. ===
    def port_conv2d(loader, keras_variable, hf_weight_key):
        loader.port_weight(
            keras_variable.kernel,
            f"{hf_weight_key}.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )
        loader.port_weight(keras_variable.bias, f"{hf_weight_key}.bias")

    def port_dense(loader, keras_variable, hf_weight_key, has_bias=True):
        loader.port_weight(
            keras_variable.kernel,
            f"{hf_weight_key}.weight",
            hook_fn=lambda x, _: x.T,
        )
        if has_bias:
            loader.port_weight(keras_variable.bias, f"{hf_weight_key}.bias")

    def port_mha(loader, keras_variable, hf_weight_key, num_heads, hidden_dim):
        for proj, hf_proj in (
            (keras_variable.query_dense, "q_proj"),
            (keras_variable.key_dense, "k_proj"),
            (keras_variable.value_dense, "v_proj"),
        ):
            loader.port_weight(
                proj.kernel,
                f"{hf_weight_key}.{hf_proj}.weight",
                hook_fn=lambda x, _: np.reshape(
                    x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
                ),
            )
            loader.port_weight(
                proj.bias,
                f"{hf_weight_key}.{hf_proj}.bias",
                hook_fn=lambda x, _: np.reshape(
                    x, (num_heads, hidden_dim // num_heads)
                ),
            )
        loader.port_weight(
            keras_variable.output_dense.kernel,
            f"{hf_weight_key}.out_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (num_heads, hidden_dim // num_heads, hidden_dim)
            ),
        )
        loader.port_weight(
            keras_variable.output_dense.bias, f"{hf_weight_key}.out_proj.bias"
        )

    def port_ln_or_gn(loader, keras_variable, hf_weight_key):
        loader.port_weight(keras_variable.gamma, f"{hf_weight_key}.weight")
        loader.port_weight(keras_variable.beta, f"{hf_weight_key}.bias")

    # === CLIP text encoder. ===
    def port_clip(preset_root, filename, model):
        with SafetensorLoader(preset_root, prefix="", fname=filename) as loader:
            embedding = model.embedding
            loader.port_weight(
                embedding.token_embedding._embeddings,
                "text_model.embeddings.token_embedding.weight",
            )
            loader.port_weight(
                embedding.position_embedding.position_embeddings,
                "text_model.embeddings.position_embedding.weight",
            )
            for i, layer in enumerate(model.encoder_layers):
                prefix = f"text_model.encoder.layers.{i}"
                port_mha(
                    loader,
                    layer.attention,
                    f"{prefix}.self_attn",
                    layer.num_heads,
                    layer.hidden_dim,
                )
                port_ln_or_gn(loader, layer.layer_norm_1, f"{prefix}.layer_norm1")
                port_ln_or_gn(loader, layer.layer_norm_2, f"{prefix}.layer_norm2")
                port_dense(loader, layer.dense_1, f"{prefix}.mlp.fc1")
                port_dense(loader, layer.dense_2, f"{prefix}.mlp.fc2")
            port_ln_or_gn(
                loader, model.layer_norm, "text_model.final_layer_norm"
            )

    # === UNet (diffusers `UNet2DConditionModel` key layout). ===
    def port_attention(loader, attn, prefix):
        # `to_q/to_k/to_v` have no bias; `to_out.0` has a bias.
        port_dense(loader, attn.to_q, f"{prefix}.to_q", has_bias=False)
        port_dense(loader, attn.to_k, f"{prefix}.to_k", has_bias=False)
        port_dense(loader, attn.to_v, f"{prefix}.to_v", has_bias=False)
        port_dense(loader, attn.to_out, f"{prefix}.to_out.0")

    def port_transformer(loader, transformer, prefix):
        port_ln_or_gn(loader, transformer.norm, f"{prefix}.norm")
        port_conv2d(loader, transformer.proj_in, f"{prefix}.proj_in")
        block = transformer.transformer_block
        block_prefix = f"{prefix}.transformer_blocks.0"
        port_ln_or_gn(loader, block.norm1, f"{block_prefix}.norm1")
        port_attention(loader, block.attn1, f"{block_prefix}.attn1")
        port_ln_or_gn(loader, block.norm2, f"{block_prefix}.norm2")
        port_attention(loader, block.attn2, f"{block_prefix}.attn2")
        port_ln_or_gn(loader, block.norm3, f"{block_prefix}.norm3")
        port_dense(loader, block.ff.proj, f"{block_prefix}.ff.net.0.proj")
        port_dense(loader, block.ff.output_dense, f"{block_prefix}.ff.net.2")
        port_conv2d(loader, transformer.proj_out, f"{prefix}.proj_out")

    def port_unet_resnet(loader, resnet, prefix):
        port_ln_or_gn(loader, resnet.norm1, f"{prefix}.norm1")
        port_conv2d(loader, resnet.conv1, f"{prefix}.conv1")
        port_dense(loader, resnet.time_emb_proj, f"{prefix}.time_emb_proj")
        port_ln_or_gn(loader, resnet.norm2, f"{prefix}.norm2")
        port_conv2d(loader, resnet.conv2, f"{prefix}.conv2")
        if resnet.conv_shortcut is not None:
            port_conv2d(loader, resnet.conv_shortcut, f"{prefix}.conv_shortcut")

    def port_unet(preset_root, filename, model):
        with SafetensorLoader(preset_root, prefix="", fname=filename) as loader:
            port_dense(
                loader, model.time_embedding.linear_1, "time_embedding.linear_1"
            )
            port_dense(
                loader, model.time_embedding.linear_2, "time_embedding.linear_2"
            )
            port_conv2d(loader, model.conv_in, "conv_in")

            # Down blocks.
            for i, block in enumerate(model.down_blocks):
                prefix = f"down_blocks.{i}"
                for j, resnet in enumerate(block.resnets):
                    port_unet_resnet(loader, resnet, f"{prefix}.resnets.{j}")
                for j, attn in enumerate(block.attentions):
                    port_transformer(loader, attn, f"{prefix}.attentions.{j}")
                if block.downsample is not None:
                    port_conv2d(
                        loader,
                        block.downsample.conv,
                        f"{prefix}.downsamplers.0.conv",
                    )

            # Mid block.
            port_unet_resnet(
                loader, model.mid_block.resnet_0, "mid_block.resnets.0"
            )
            port_transformer(
                loader, model.mid_block.attention, "mid_block.attentions.0"
            )
            port_unet_resnet(
                loader, model.mid_block.resnet_1, "mid_block.resnets.1"
            )

            # Up blocks.
            for i, block in enumerate(model.up_blocks):
                prefix = f"up_blocks.{i}"
                for j, resnet in enumerate(block.resnets):
                    port_unet_resnet(loader, resnet, f"{prefix}.resnets.{j}")
                for j, attn in enumerate(block.attentions):
                    port_transformer(loader, attn, f"{prefix}.attentions.{j}")
                if block.upsample is not None:
                    port_conv2d(
                        loader,
                        block.upsample.conv,
                        f"{prefix}.upsamplers.0.conv",
                    )

            port_ln_or_gn(loader, model.conv_norm_out, "conv_norm_out")
            port_conv2d(loader, model.conv_out, "conv_out")

    # === VAE (diffusers `AutoencoderKL` key layout). ===
    def port_vae_resnet(loader, resnet, prefix):
        port_ln_or_gn(loader, resnet.norm1, f"{prefix}.norm1")
        port_conv2d(loader, resnet.conv1, f"{prefix}.conv1")
        port_ln_or_gn(loader, resnet.norm2, f"{prefix}.norm2")
        port_conv2d(loader, resnet.conv2, f"{prefix}.conv2")
        if hasattr(resnet, "residual_projection"):
            port_conv2d(
                loader, resnet.residual_projection, f"{prefix}.conv_shortcut"
            )

    def port_vae_attention(loader, attn, prefix):
        port_ln_or_gn(loader, attn.group_norm, f"{prefix}.group_norm")
        # CompVis SD1.4 was saved with the legacy `AttentionBlock` key names;
        # newer diffusers re-saves use `to_q`/`to_k`/`to_v`/`to_out.0`.
        try:
            loader.get_tensor(f"{prefix}.to_q.weight")
            q_key, k_key, v_key, out_key = "to_q", "to_k", "to_v", "to_out.0"
        except Exception:
            q_key, k_key, v_key, out_key = (
                "query",
                "key",
                "value",
                "proj_attn",
            )

        # The projections are stored as 2D linear weights (or, in some
        # checkpoints, as 1x1 convs); reshape either to a keras 1x1 conv kernel.
        def to_conv(x, _):
            if x.ndim == 2:
                x = x[:, :, None, None]
            return x.transpose(2, 3, 1, 0)

        for keras_conv, hf_key in (
            (attn.query_conv2d, q_key),
            (attn.key_conv2d, k_key),
            (attn.value_conv2d, v_key),
            (attn.output_conv2d, out_key),
        ):
            loader.port_weight(
                keras_conv.kernel, f"{prefix}.{hf_key}.weight", hook_fn=to_conv
            )
            loader.port_weight(keras_conv.bias, f"{prefix}.{hf_key}.bias")

    def port_vae(preset_root, filename, model):
        with SafetensorLoader(preset_root, prefix="", fname=filename) as loader:
            # Encoder.
            encoder = model.encoder
            port_conv2d(loader, encoder.input_projection, "encoder.conv_in")
            blocks_idx = 0
            downsamples_idx = 0
            for i, _ in enumerate(encoder.stackwise_num_filters):
                for j in range(encoder.stackwise_num_blocks[i]):
                    port_vae_resnet(
                        loader,
                        encoder.blocks[blocks_idx],
                        f"encoder.down_blocks.{i}.resnets.{j}",
                    )
                    blocks_idx += 1
                if i != len(encoder.stackwise_num_filters) - 1:
                    # Skip the `ZeroPadding2D` at `downsamples_idx`.
                    port_conv2d(
                        loader,
                        encoder.downsamples[downsamples_idx + 1],
                        f"encoder.down_blocks.{i}.downsamplers.0.conv",
                    )
                    downsamples_idx += 2
            port_vae_resnet(
                loader, encoder.mid_block_0, "encoder.mid_block.resnets.0"
            )
            port_vae_attention(
                loader, encoder.mid_attention, "encoder.mid_block.attentions.0"
            )
            port_vae_resnet(
                loader, encoder.mid_block_1, "encoder.mid_block.resnets.1"
            )
            port_ln_or_gn(loader, encoder.output_norm, "encoder.conv_norm_out")
            # Fold `quant_conv` (1x1, applied AFTER `conv_out`) into the encoder
            # output projection. Exact: a 1x1 conv after a 3x3 conv composes
            # cleanly into the 3x3 weights and bias.
            qw = loader.get_tensor("quant_conv.weight")[:, :, 0, 0]  # [8, 8]
            qb = loader.get_tensor("quant_conv.bias")  # [8]
            loader.port_weight(
                encoder.output_projection.kernel,
                "encoder.conv_out.weight",
                hook_fn=lambda w, _: np.transpose(
                    np.einsum("oc,cijk->oijk", qw, w), (2, 3, 1, 0)
                ),
            )
            loader.port_weight(
                encoder.output_projection.bias,
                "encoder.conv_out.bias",
                hook_fn=lambda b, _: np.einsum("oc,c->o", qw, b) + qb,
            )

            # Decoder.
            decoder = model.decoder
            # Fold `post_quant_conv` (1x1, applied BEFORE `conv_in`) into the
            # decoder input projection. The weight fold is exact; the bias fold
            # is exact in the interior but approximate on the 1-pixel latent
            # border (the padded 3x3 conv of a constant bias field is not
            # spatially constant at the edges).
            pw = loader.get_tensor("post_quant_conv.weight")[:, :, 0, 0]  # [4,4]
            pb = loader.get_tensor("post_quant_conv.bias")  # [4]
            loader.port_weight(
                decoder.input_projection.kernel,
                "decoder.conv_in.weight",
                hook_fn=lambda w, _: np.transpose(
                    np.einsum("ocjk,ci->oijk", w, pw), (2, 3, 1, 0)
                ),
            )
            loader.port_weight(
                decoder.input_projection.bias,
                "decoder.conv_in.bias",
                hook_fn=lambda b, _: b
                + np.einsum("ocjk,c->o", loader.get_tensor(
                    "decoder.conv_in.weight"
                ), pb),
            )
            port_vae_resnet(
                loader, decoder.mid_block_0, "decoder.mid_block.resnets.0"
            )
            port_vae_attention(
                loader, decoder.mid_attention, "decoder.mid_block.attentions.0"
            )
            port_vae_resnet(
                loader, decoder.mid_block_1, "decoder.mid_block.resnets.1"
            )
            blocks_idx = 0
            upsamples_idx = 0
            # Unlike the CompVis single-file layout, diffusers numbers the
            # decoder `up_blocks` in the same order as the keras decoder stacks.
            for i, _ in enumerate(decoder.stackwise_num_filters):
                for j in range(decoder.stackwise_num_blocks[i]):
                    port_vae_resnet(
                        loader,
                        decoder.blocks[blocks_idx],
                        f"decoder.up_blocks.{i}.resnets.{j}",
                    )
                    blocks_idx += 1
                if i != len(decoder.stackwise_num_filters) - 1:
                    # Skip the `UpSampling2D` at `upsamples_idx`.
                    port_conv2d(
                        loader,
                        decoder.upsamples[upsamples_idx + 1],
                        f"decoder.up_blocks.{i}.upsamplers.0.conv",
                    )
                    upsamples_idx += 2
            port_ln_or_gn(loader, decoder.output_norm, "decoder.conv_norm_out")
            port_conv2d(loader, decoder.output_projection, "decoder.conv_out")

    # Start conversion.
    port_clip(config["root"], config["clip"], keras_model.clip)
    port_unet(config["root"], config["unet"], keras_model.diffuser)
    port_vae(config["root"], config["vae"], keras_model.vae)


def validate_output(preset, keras_model, keras_preprocessor, output_dir):
    config = PRESET_MAP[preset]
    dtype = config["dtype"]
    prompt = "A cat holding a sign that says hello world"
    num_steps = 25
    guidance_scale = 7.5

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, torch.float32)
    hf_repo_id = config["root"].replace("hf://", "", 1)
    pipeline = StableDiffusionPipeline.from_pretrained(
        hf_repo_id, torch_dtype=torch_dtype, safety_checker=None
    )
    text_to_image = StableDiffusion1_4TextToImage(
        keras_model, keras_preprocessor
    )

    # Parameter count.
    hf_params = (
        sum(p.numel() for p in pipeline.unet.parameters())
        + sum(p.numel() for p in pipeline.vae.parameters())
        + sum(p.numel() for p in pipeline.text_encoder.parameters())
    )
    print("🔶 Keras params:", keras_model.count_params())
    print("🔶 HF params:   ", hf_params)

    # Text encoder.
    token_ids = text_to_image.preprocessor.generate_preprocess([prompt])
    with torch.inference_mode():
        hf_text_input = pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        hf_embeddings = (
            pipeline.text_encoder(hf_text_input.input_ids)[0]
            .to(torch.float32)
            .numpy()
        )
    keras_embeddings, _ = text_to_image.backbone.encode_text_step(
        token_ids, token_ids
    )
    keras_embeddings = keras.ops.convert_to_numpy(
        keras.ops.cast(keras_embeddings, "float32")
    )
    diff = np.abs(keras_embeddings - hf_embeddings)
    print("🔶 Text embeddings diff (mean/max):", diff.mean(), diff.max())

    # VAE round trip. The VAE expects images normalized to `[-1, 1]`; feed the
    # same normalized input to both models.
    np.random.seed(0)
    images = np.random.rand(1, 512, 512, 3).astype("float32") * 2.0 - 1.0
    keras_latents = text_to_image.backbone.encode_image_step(images)
    keras_decoded = text_to_image.backbone.decode_step(keras_latents)
    with torch.inference_mode():
        hf_in = torch.from_numpy(images.transpose(0, 3, 1, 2))
        hf_latents = (
            pipeline.vae.encode(hf_in.to(torch_dtype)).latent_dist.mode()
            * pipeline.vae.config.scaling_factor
        )
        hf_decoded = pipeline.vae.decode(
            hf_latents / pipeline.vae.config.scaling_factor
        ).sample
    hf_decoded = hf_decoded.to(torch.float32).numpy().transpose(0, 2, 3, 1)
    vae_diff = np.abs(
        keras.ops.convert_to_numpy(
            keras.ops.cast(keras_decoded, "float32")
        )
        - hf_decoded
    )
    print("🔶 VAE decode diff (mean/max):", vae_diff.mean(), vae_diff.max())

    # Generate an image.
    image = text_to_image.generate(
        prompt, num_steps=num_steps, guidance_scale=guidance_scale, seed=42
    )
    Image.fromarray(image).save(os.path.join(output_dir, f"{preset}.png"))


def main(_):
    preset = FLAGS.preset
    output_dir = FLAGS.output_dir
    if os.path.exists(preset):
        shutil.rmtree(preset)
    os.makedirs(preset, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"🏃 Converting {preset}")
    keras.config.set_dtype_policy(PRESET_MAP[preset]["dtype"])
    height, width = 512, 512

    keras_preprocessor = convert_preprocessor(preset)
    keras_model = convert_model(preset, height, width)
    print("✅ KerasHub model loaded.")

    convert_weights(preset, keras_model)
    print("✅ Weights converted.")

    validate_output(preset, keras_model, keras_preprocessor, output_dir)
    print("✅ Output validated.")

    keras_preprocessor.save_to_preset(preset)
    keras_model.save_to_preset(preset)
    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
