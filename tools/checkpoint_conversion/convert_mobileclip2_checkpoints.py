"""Convert MobileCLIP2 checkpoints.

MobileCLIP2 weights are distributed by Apple as open_clip `.pt` checkpoints
(not HuggingFace `transformers` `CLIPModel`), so this script loads the reference
model with `open_clip` and ports the state dict.

Requirements: `pip install open_clip_torch timm torch`.

export KAGGLE_USERNAME=XXX
export KAGGLE_KEY=XXX

python tools/checkpoint_conversion/convert_mobileclip2_checkpoints.py \
    --preset mobileclip2_b \
    --upload_uri kaggle://kerashub/mobileclip2/keras/mobileclip2_b
"""

import json
import os
import shutil

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from huggingface_hub import hf_hub_download
from PIL import Image

import keras_hub
from keras_hub.src.models.mobileclip2.mobileclip2_backbone import (
    MobileCLIP2Backbone,
)
from keras_hub.src.models.mobileclip2.mobileclip2_image_converter import (
    MobileCLIP2ImageConverter,
)
from keras_hub.src.models.mobileclip2.mobileclip2_preprocessor import (
    MobileCLIP2Preprocessor,
)
from keras_hub.src.models.mobileclip2.mobileclip2_text_encoder import (
    MobileCLIP2TextEncoder,
)
from keras_hub.src.models.mobileclip2.mobileclip2_tokenizer import (
    MobileCLIP2Tokenizer,
)
from keras_hub.src.models.mobileclip2.mobileclip2_vision_encoder import (
    MobileCLIP2VisionEncoder,
)

FLAGS = flags.FLAGS

# preset -> (hf_repo_id, checkpoint_filename, open_clip_model_name)
PRESET_MAP = {
    "mobileclip2_b": (
        "apple/MobileCLIP2-B",
        "mobileclip2_b.pt",
        "MobileCLIP-B",
    ),
}
# CLIP BPE vocabulary/merges live in this public repo's `tokenizer.json`.
TOKENIZER_REPO = "openai/clip-vit-base-patch32"

flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
    required=True,
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://kerashub/mobileclip2/keras/{preset}"',
    required=False,
)
flags.DEFINE_bool(
    "print_keys",
    False,
    "Print the reference checkpoint's state_dict keys and exit.",
)


def build_keras_model():
    """Build the randomly initialized MobileCLIP2-B backbone."""
    vision_encoder = MobileCLIP2VisionEncoder(
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        intermediate_dim=3072,
        intermediate_activation="gelu",
        stem_channels=(192, 192, 768),
        stem_kernel_sizes=(4, 2, 2),
        stem_strides=(4, 2, 2),
        image_shape=(224, 224, 3),
    )
    text_encoder = MobileCLIP2TextEncoder(
        vocabulary_size=49408,
        embedding_dim=512,
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        intermediate_dim=2048,
        intermediate_activation="gelu",
        max_sequence_length=77,
    )
    return MobileCLIP2Backbone(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        projection_dim=512,
    )


def convert_weights(keras_model, state_dict):
    # Helper functions.
    def port_weights(keras_variable, weight_key, hook_fn=None):
        torch_tensor = state_dict[weight_key].cpu().numpy()
        if hook_fn:
            torch_tensor = hook_fn(torch_tensor, list(keras_variable.shape))
        keras_variable.assign(torch_tensor)

    def port_ln(keras_variable, weight_key):
        port_weights(keras_variable.gamma, f"{weight_key}.weight")
        port_weights(keras_variable.beta, f"{weight_key}.bias")

    def port_bn(keras_variable, weight_key):
        port_weights(keras_variable.gamma, f"{weight_key}.weight")
        port_weights(keras_variable.beta, f"{weight_key}.bias")
        port_weights(keras_variable.moving_mean, f"{weight_key}.running_mean")
        port_weights(
            keras_variable.moving_variance, f"{weight_key}.running_var"
        )

    def port_dense(keras_variable, weight_key):
        port_weights(
            keras_variable.kernel,
            f"{weight_key}.weight",
            hook_fn=lambda x, _: x.T,
        )
        if keras_variable.bias is not None:
            port_weights(keras_variable.bias, f"{weight_key}.bias")

    def port_conv(keras_variable, weight_key):
        port_weights(
            keras_variable.kernel,
            f"{weight_key}.weight",
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )
        if keras_variable.bias is not None:
            port_weights(keras_variable.bias, f"{weight_key}.bias")

    def port_fused_mha(
        keras_attn, fused_weight_key, fused_bias_key, out_key, num_heads, dim
    ):
        head_dim = dim // num_heads
        fused_weight = (
            state_dict[fused_weight_key].cpu().numpy()
        )  # (3*dim, dim)
        q_w, k_w, v_w = np.split(fused_weight, 3, axis=0)
        fused_bias = state_dict[fused_bias_key].cpu().numpy()  # (3*dim,)
        q_b, k_b, v_b = np.split(fused_bias, 3, axis=0)
        for dense, w, b in (
            (keras_attn.query_dense, q_w, q_b),
            (keras_attn.key_dense, k_w, k_b),
            (keras_attn.value_dense, v_w, v_b),
        ):
            dense.kernel.assign(np.reshape(w.T, (dim, num_heads, head_dim)))
            dense.bias.assign(np.reshape(b, (num_heads, head_dim)))
        port_weights(
            keras_attn.output_dense.kernel,
            f"{out_key}.weight",
            hook_fn=lambda x, _: np.reshape(x.T, (num_heads, head_dim, dim)),
        )
        port_weights(keras_attn.output_dense.bias, f"{out_key}.bias")

    # === Vision encoder (timm `vit_base_mci_224`, prefix `visual.trunk`) ===
    vision_encoder = keras_model.vision_encoder
    embedding = vision_encoder.embedding
    port_weights(
        embedding.class_embedding,
        "visual.trunk.cls_token",
        hook_fn=lambda x, _: np.reshape(x, (-1,)),
    )
    port_weights(
        embedding.position_embedding.embeddings,
        "visual.trunk.pos_embed",
        hook_fn=lambda x, _: x[0],
    )
    # Convolutional patch stem.
    stem = embedding.conv_stem
    for i, conv in enumerate(stem.convs):
        port_conv(conv, f"visual.trunk.patch_embed.backbone.{i}.conv")
        if stem.norms[i] is not None:
            port_bn(stem.norms[i], f"visual.trunk.patch_embed.backbone.{i}.bn")
    # Transformer blocks.
    for i, block in enumerate(vision_encoder.encoder_layers):
        prefix = f"visual.trunk.blocks.{i}"
        port_ln(block.layer_norm_1, f"{prefix}.norm1")
        port_fused_mha(
            block.attention,
            f"{prefix}.attn.qkv.weight",
            f"{prefix}.attn.qkv.bias",
            f"{prefix}.attn.proj",
            block.num_heads,
            block.hidden_dim,
        )
        port_ln(block.layer_norm_2, f"{prefix}.norm2")
        port_dense(block.dense_1, f"{prefix}.mlp.fc1")
        port_dense(block.dense_2, f"{prefix}.mlp.fc2")
    port_ln(vision_encoder.layer_norm, "visual.trunk.norm")
    # Image projection == timm classifier head `Linear(768, 512)` (with bias).
    port_dense(keras_model.vision_projection, "visual.trunk.head")

    # === Text encoder (open_clip TextTransformer, prefix `text`) ===
    text_encoder = keras_model.text_encoder
    port_weights(
        text_encoder.embedding.token_embedding._embeddings,
        "text.token_embedding.weight",
    )
    port_weights(
        text_encoder.embedding.position_embedding.position_embeddings,
        "text.positional_embedding",
    )
    for i, block in enumerate(text_encoder.encoder_layers):
        prefix = f"text.transformer.resblocks.{i}"
        port_ln(block.layer_norm_1, f"{prefix}.ln_1")
        port_fused_mha(
            block.attention,
            f"{prefix}.attn.in_proj_weight",
            f"{prefix}.attn.in_proj_bias",
            f"{prefix}.attn.out_proj",
            block.num_heads,
            block.hidden_dim,
        )
        port_ln(block.layer_norm_2, f"{prefix}.ln_2")
        port_dense(block.dense_1, f"{prefix}.mlp.c_fc")
        port_dense(block.dense_2, f"{prefix}.mlp.c_proj")
    port_ln(text_encoder.layer_norm, "text.ln_final")
    # Text projection is stored either as a bare `nn.Parameter` of shape
    # `(width, embed_dim)` or a bias-less `nn.Linear` (`text_projection.weight`
    # of shape `(embed_dim, width)`).
    if "text.text_projection" in state_dict:
        port_weights(keras_model.text_projection.kernel, "text.text_projection")
    else:
        port_weights(
            keras_model.text_projection.kernel,
            "text.text_projection.weight",
            hook_fn=lambda x, _: x.T,
        )

    # Logit scale.
    port_weights(keras_model.mobileclip2_head.logit_scale, "logit_scale")


def convert_image_converter():
    # MobileCLIP2-B normalizes with mean=(0,0,0), std=(1,1,1): only `[0, 1]`
    # scaling, unlike the OpenCLIP-normalized S3/S4/L variants.
    return MobileCLIP2ImageConverter(
        image_size=(224, 224),
        scale=1.0 / 255.0,
        offset=0.0,
        interpolation="bicubic",
    )


def convert_tokenizer():
    tokenizer_path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json")
    with open(tokenizer_path, "r") as tokenizer_file:
        tokenizer_content = json.load(tokenizer_file)
    vocabulary = tokenizer_content["model"]["vocab"]
    merges = tokenizer_content["model"]["merges"]
    return MobileCLIP2Tokenizer(
        vocabulary,
        merges,
        pad_with_end_token=True,
    )


def validate_output(
    keras_model,
    keras_image_converter,
    keras_tokenizer,
    reference_model,
    open_clip,
):
    file = keras.utils.get_file(
        origin=("http://images.cocodataset.org/val2017/000000039769.jpg")
    )
    image = Image.open(file).convert("RGB")
    text = ["a photo of a cat", "a photo of a dog"]

    # Preprocess the image with keras and reuse the tensor for the reference
    # model so that modeling and preprocessing comparisons stay independent.
    images = np.expand_dims(np.array(image).astype("float32"), axis=0)
    keras_preprocessed = keras.ops.convert_to_numpy(
        keras_image_converter(images)
    )

    # Reference (open_clip) forward pass.
    reference_tokenizer = open_clip.get_tokenizer("MobileCLIP-B")
    reference_model.eval()
    with torch.no_grad():
        pixel_values = torch.from_numpy(
            np.transpose(keras_preprocessed, (0, 3, 1, 2))
        )
        image_features = reference_model.encode_image(pixel_values)
        text_features = reference_model.encode_text(reference_tokenizer(text))
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = reference_model.logit_scale.exp()
        reference_vision_logits = (
            (logit_scale * image_features @ text_features.t()).cpu().numpy()
        )

    # Keras forward pass.
    keras_preprocessor = MobileCLIP2Preprocessor(keras_tokenizer)
    token_ids = keras_preprocessor(
        {"images": keras_preprocessed, "prompts": text}
    )["token_ids"]
    keras_outputs = keras_model.predict(
        {"images": keras_preprocessed, "token_ids": token_ids}, verbose=0
    )
    keras_vision_logits = keras.ops.convert_to_numpy(
        keras_outputs["vision_logits"]
    )

    print("🔶 Keras output:", keras_vision_logits[0])
    print("🔶 Reference output:", reference_vision_logits[0])
    modeling_diff = np.mean(
        np.abs(keras_vision_logits - reference_vision_logits)
    )
    print("🔶 Modeling difference:", modeling_diff)


def main(_):
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    import open_clip

    preset = FLAGS.preset
    hf_repo, checkpoint_filename, open_clip_name = PRESET_MAP[preset]

    print(f"🏃 Downloading {hf_repo}/{checkpoint_filename}")
    checkpoint_path = hf_hub_download(hf_repo, checkpoint_filename)
    reference_model, _, _ = open_clip.create_model_and_transforms(
        open_clip_name, pretrained=checkpoint_path
    )
    reference_model.eval()
    state_dict = reference_model.state_dict()

    if FLAGS.print_keys:
        for key, value in state_dict.items():
            print(key, tuple(value.shape))
        return

    if os.path.exists(preset):
        shutil.rmtree(preset)
    os.makedirs(preset)

    print(f"🏃 Converting {preset}")
    keras_model = build_keras_model()
    keras_image_converter = convert_image_converter()
    keras_tokenizer = convert_tokenizer()
    print("✅ KerasHub model loaded.")

    convert_weights(keras_model, state_dict)
    print("✅ Weights converted.")

    validate_output(
        keras_model,
        keras_image_converter,
        keras_tokenizer,
        reference_model,
        open_clip,
    )
    print("✅ Output validated.")

    keras_model.save_to_preset(f"./{preset}")
    keras_image_converter.save_to_preset(f"./{preset}")
    keras_tokenizer.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
