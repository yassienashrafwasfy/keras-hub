"""Convert Stable Diffusion v1.4 checkpoints.

python tools/checkpoint_conversion/convert_stable_diffusion_1_4_checkpoints.py \
    --preset stable_diffusion_1_4 \
    --upload_uri kaggle://kerashub/stablediffusion1.4/keras/stable_diffusion_1_4
"""

import os

# Use the torch backend for conversion. Must be set before `keras` is imported.
os.environ["KERAS_BACKEND"] = "torch"

import gc  # noqa: E402
import shutil  # noqa: E402

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from diffusers import AutoencoderKL  # noqa: E402
from diffusers import DPMSolverMultistepScheduler  # noqa: E402
from diffusers import StableDiffusionPipeline  # noqa: E402
from diffusers import UNet2DConditionModel  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import CLIPTextModel  # noqa: E402

import keras_hub  # noqa: E402
from keras_hub.src.models.clip.clip_preprocessor import (  # noqa: E402
    CLIPPreprocessor,
)
from keras_hub.src.models.clip.clip_text_encoder import (  # noqa: E402
    CLIPTextEncoder,
)
from keras_hub.src.models.clip.clip_tokenizer import CLIPTokenizer  # noqa: E402
from keras_hub.src.models.stable_diffusion_1_4.stable_diffusion_1_4_backbone import (  # noqa: E402, E501
    StableDiffusion1_4Backbone,
)
from keras_hub.src.models.stable_diffusion_1_4.stable_diffusion_1_4_text_to_image import (  # noqa: E402, E501
    StableDiffusion1_4TextToImage,
)
from keras_hub.src.models.stable_diffusion_1_4.stable_diffusion_1_4_text_to_image_preprocessor import (  # noqa: E402, E501
    StableDiffusion1_4TextToImagePreprocessor,
)
from keras_hub.src.models.vae.vae_backbone import VAEBackbone  # noqa: E402
from keras_hub.src.utils.preset_utils import load_json  # noqa: E402
from keras_hub.src.utils.transformers.safetensor_utils import (  # noqa: E402
    SafetensorLoader,
)

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
            port_conv2d(loader, encoder.output_projection, "encoder.conv_out")

            # Decoder.
            decoder = model.decoder
            port_conv2d(loader, decoder.input_projection, "decoder.conv_in")
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
    # `quant_conv` / `post_quant_conv` live on the backbone, not the VAE.
    with SafetensorLoader(
        config["root"], prefix="", fname=config["vae"]
    ) as loader:
        port_conv2d(loader, keras_model.quant_conv, "quant_conv")
        port_conv2d(loader, keras_model.post_quant_conv, "post_quant_conv")


def validate_output(preset, keras_model, keras_preprocessor, output_dir):
    # Inference only: disable autograd so the multi-step keras generation does
    # not build a graph through the UNet for every diffusion step.
    torch.set_grad_enabled(False)

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

    text_to_image = StableDiffusion1_4TextToImage(
        keras_model, keras_preprocessor
    )
    token_ids = text_to_image.preprocessor.generate_preprocess([prompt])

    # Each HF component is loaded, compared, then freed before the next, so the
    # keras model and a full HF model are never resident at the same time.
    hf_param_count = 0

    # === Text encoder. Feed identical token ids to both to isolate it. ===
    token_ids_np = keras.ops.convert_to_numpy(token_ids).astype("int64")
    hf_text_encoder = CLIPTextModel.from_pretrained(
        hf_repo_id, subfolder="text_encoder", torch_dtype=torch_dtype
    )
    hf_param_count += sum(p.numel() for p in hf_text_encoder.parameters())
    hf_embeddings = (
        hf_text_encoder(torch.from_numpy(token_ids_np))[0]
        .to(torch.float32)
        .numpy()
    )
    del hf_text_encoder
    gc.collect()
    keras_embeddings, _ = text_to_image.backbone.encode_text_step(
        token_ids, token_ids
    )
    keras_embeddings = keras.ops.convert_to_numpy(
        keras.ops.cast(keras_embeddings, "float32")
    )
    diff = np.abs(keras_embeddings - hf_embeddings)
    print("🔶 Text embeddings diff (mean/max):", diff.mean(), diff.max())

    # === UNet. Feed identical latent, timestep and embeddings to both. ===
    np.random.seed(1)
    latent = np.random.randn(1, 4, 64, 64).astype("float32")
    timestep = 500
    hf_unet = UNet2DConditionModel.from_pretrained(
        hf_repo_id, subfolder="unet", torch_dtype=torch_dtype
    )
    hf_param_count += sum(p.numel() for p in hf_unet.parameters())
    hf_eps = (
        hf_unet(
            torch.from_numpy(latent).to(torch_dtype),
            torch.tensor(timestep),
            encoder_hidden_states=torch.from_numpy(hf_embeddings).to(
                torch_dtype
            ),
        )
        .sample.to(torch.float32)
        .numpy()
        .transpose(0, 2, 3, 1)
    )
    del hf_unet
    gc.collect()
    keras_eps = text_to_image.backbone.diffuser(
        {
            "latent": keras.ops.convert_to_tensor(latent.transpose(0, 2, 3, 1)),
            "timestep": keras.ops.convert_to_tensor([float(timestep)]),
            "context": keras.ops.convert_to_tensor(hf_embeddings),
        }
    )
    keras_eps = keras.ops.convert_to_numpy(keras.ops.cast(keras_eps, "float32"))
    unet_diff = np.abs(keras_eps - hf_eps)
    print("🔶 UNet eps diff (mean/max):", unet_diff.mean(), unet_diff.max())

    # === VAE round trip. The VAE expects images normalized to [-1, 1]. ===
    np.random.seed(0)
    images = np.random.rand(1, 512, 512, 3).astype("float32") * 2.0 - 1.0
    keras_latents = text_to_image.backbone.encode_image_step(images)
    keras_decoded = text_to_image.backbone.decode_step(keras_latents)
    hf_vae = AutoencoderKL.from_pretrained(
        hf_repo_id, subfolder="vae", torch_dtype=torch_dtype
    )
    hf_param_count += sum(p.numel() for p in hf_vae.parameters())
    hf_in = torch.from_numpy(images.transpose(0, 3, 1, 2)).to(torch_dtype)
    hf_latents = (
        hf_vae.encode(hf_in).latent_dist.mode() * hf_vae.config.scaling_factor
    )
    hf_decoded = (
        hf_vae.decode(hf_latents / hf_vae.config.scaling_factor)
        .sample.to(torch.float32)
        .numpy()
        .transpose(0, 2, 3, 1)
    )
    del hf_vae
    gc.collect()
    vae_diff = np.abs(
        keras.ops.convert_to_numpy(keras.ops.cast(keras_decoded, "float32"))
        - hf_decoded
    )
    print("🔶 VAE decode diff (mean/max):", vae_diff.mean(), vae_diff.max())

    # === Parameter count. ===
    print("🔶 Keras params:", keras_model.count_params())
    print("🔶 HF params:   ", hf_param_count)

    # === End-to-end image from identical initial latents. ===
    # The full pipeline is freed before the keras generation so both full
    # models are never resident together.
    pipeline = StableDiffusionPipeline.from_pretrained(
        hf_repo_id, torch_dtype=torch_dtype, safety_checker=None
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        algorithm_type="dpmsolver++",
        solver_order=2,
        use_karras_sigmas=True,
    )
    np.random.seed(42)
    init_latents = np.random.randn(1, 4, 64, 64).astype("float32")
    hf_image = pipeline(
        prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        latents=torch.from_numpy(init_latents).to(torch_dtype),
        output_type="np",
    ).images[0]
    hf_image = (hf_image * 255).round().astype("uint8")
    del pipeline
    gc.collect()

    text_to_image.backbone.configure_scheduler(num_steps)
    negative_token_ids = text_to_image.preprocessor.generate_preprocess([""])
    keras_decoded = text_to_image.generate_step(
        keras.ops.convert_to_tensor(init_latents.transpose(0, 2, 3, 1)),
        (token_ids, negative_token_ids),
        num_steps,
        keras.ops.convert_to_tensor(float(guidance_scale)),
    )
    keras_image = keras.ops.convert_to_numpy(
        keras.ops.cast(
            keras.ops.clip(
                (keras.ops.cast(keras_decoded, "float32") + 1.0) / 2.0,
                0.0,
                1.0,
            )
            * 255.0,
            "uint8",
        )
    )[0]
    image_diff = np.abs(
        keras_image.astype("float32") - hf_image.astype("float32")
    )
    print("🔶 Image diff (mean/max):", image_diff.mean(), image_diff.max())
    # Save the two images side by side (keras | diffusers).
    combined = np.concatenate([keras_image, hf_image], axis=1)
    Image.fromarray(combined).save(os.path.join(output_dir, f"{preset}.png"))


def main(_):
    preset = FLAGS.preset
    output_dir = FLAGS.output_dir
    if os.path.exists(preset):
        shutil.rmtree(preset)
    os.makedirs(preset, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"🏃 Converting {preset}")
    # Conversion is inference only; never accumulate autograd state.
    torch.set_grad_enabled(False)
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
