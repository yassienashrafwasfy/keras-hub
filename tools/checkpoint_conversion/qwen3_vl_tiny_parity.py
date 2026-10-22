"""Compare KerasHub Qwen3-VL against tiny random-init HF models.

Builds seeded tiny `Qwen3VLForConditionalGeneration` models (tied and untied
embeddings), saves them to a local temp dir, loads them through the KerasHub
HF converter and compares vision features, DeepStack features, per-layer
hidden states and logits on CPU float32. Nothing is downloaded.

Usage:
    python tools/checkpoint_conversion/qwen3_vl_tiny_parity.py
"""

import argparse
import os
import sys
import tempfile

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import keras
import numpy as np
import torch
from transformers import Qwen3VLConfig
from transformers import Qwen3VLForConditionalGeneration

import keras_hub

IMAGE_TOKEN_ID = 250
VIDEO_TOKEN_ID = 251
VISION_START_ID = 252
VISION_END_ID = 253
VOCAB_SIZE = 300
DEPTH = 4
DEEPSTACK_INDEXES = [1, 2]
MERGE = 2


def build_config(tie_word_embeddings):
    return Qwen3VLConfig(
        text_config=dict(
            vocab_size=VOCAB_SIZE,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=DEPTH,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rope_parameters=dict(
                rope_type="default",
                rope_theta=10000.0,
                mrope_section=[4, 2, 2],
                mrope_interleaved=True,
            ),
        ),
        vision_config=dict(
            depth=DEPTH,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=64,
            num_position_embeddings=64,
            deepstack_visual_indexes=DEEPSTACK_INDEXES,
        ),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID,
        tie_word_embeddings=tie_word_embeddings,
    )


def randomize_(model, seed):
    # Default init leaves norms at 1 and biases at 0, which hides mapping bugs.
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            noise = torch.randn(param.shape, generator=generator)
            if param.ndim == 1 and name.endswith("norm.weight"):
                param.copy_(1.0 + 0.1 * noise)
            elif param.ndim == 1:
                param.copy_(0.05 * noise)
            else:
                param.copy_(0.1 * noise)
    if model.config.tie_word_embeddings:
        model.tie_weights()


def make_inputs(kind, seed):
    """Return HF-side tensors and the flat positions of vision tokens."""
    rng = np.random.RandomState(seed)

    def text(n):
        return rng.randint(0, 240, size=n).tolist()

    ids, types = text(3), [0] * 3
    result = {}
    if kind == "image":
        grid = np.array([[1, 4, 4]])
        n_tokens = int(grid[0].prod()) // MERGE**2
        ids += [VISION_START_ID] + [IMAGE_TOKEN_ID] * n_tokens
        ids += [VISION_END_ID] + text(2)
        types += [0] + [1] * n_tokens + [0] * 3
        key = "pixel_values"
    elif kind == "video":
        grid = np.array([[2, 4, 4]])
        per_frame = int(grid[0, 1] * grid[0, 2]) // MERGE**2
        for _ in range(int(grid[0, 0])):
            ids += text(2) + [VISION_START_ID]
            ids += [VIDEO_TOKEN_ID] * per_frame + [VISION_END_ID]
            types += [0] * 3 + [2] * per_frame + [0]
        ids += text(2)
        types += [0] * 2
        key = "pixel_values_videos"
    else:
        ids += text(9)
        types += [0] * 9
        grid = None
    result["input_ids"] = torch.tensor([ids])
    result["mm_token_type_ids"] = torch.tensor([types], dtype=torch.int32)
    result["attention_mask"] = torch.ones_like(result["input_ids"])
    if grid is not None:
        patch_dim = 3 * 2 * 16 * 16
        n_patches = int(grid[0].prod())
        result[key] = torch.tensor(
            rng.randn(n_patches, patch_dim).astype("float32")
        )
        result["video_grid_thw" if kind == "video" else "image_grid_thw"] = (
            torch.tensor(grid)
        )
    vision_positions = np.where(np.array(types) != 0)[0].astype("int32")
    return result, vision_positions


def hf_reference(model, kind, hf_inputs):
    layer_outputs = []
    hooks = [
        layer.register_forward_hook(
            lambda _m, _i, out: layer_outputs.append(
                (out[0] if isinstance(out, tuple) else out).detach().numpy()
            )
        )
        for layer in model.model.language_model.layers
    ]
    with torch.no_grad():
        logits = model(**hf_inputs).logits.numpy()
        reference = {}
        pixels_key = {
            "image": ("pixel_values", "image_grid_thw"),
            "video": ("pixel_values_videos", "video_grid_thw"),
        }.get(kind)
        if pixels_key is not None:
            vision = model.model.visual(
                hf_inputs[pixels_key[0]], grid_thw=hf_inputs[pixels_key[1]]
            )
            reference["vision"] = vision.pooler_output.numpy()
            for i, feat in enumerate(vision.deepstack_features):
                reference[f"deepstack_{i}"] = feat.numpy()
        reference["position_ids"] = (
            model.model.get_rope_index(
                hf_inputs["input_ids"],
                hf_inputs["mm_token_type_ids"],
                image_grid_thw=hf_inputs.get("image_grid_thw"),
                video_grid_thw=(
                    hf_inputs["video_grid_thw"].clone()
                    if "video_grid_thw" in hf_inputs
                    else None
                ),
            )[0]
            .permute(1, 0, 2)
            .numpy()
            .astype("int32")
        )
    for hook in hooks:
        hook.remove()
    for i, out in enumerate(layer_outputs):
        reference[f"layer_{i}"] = out
    reference["logits"] = logits
    return reference


def keras_inputs(backbone, kind, hf_inputs, vision_positions):
    token_ids = hf_inputs["input_ids"].numpy().astype("int32")
    seq_len = token_ids.shape[1]
    inputs = {
        "token_ids": token_ids,
        "padding_mask": np.ones_like(token_ids),
    }
    pixels = hf_inputs.get("pixel_values", hf_inputs.get("pixel_values_videos"))
    grid = hf_inputs.get("image_grid_thw", hf_inputs.get("video_grid_thw"))
    if pixels is not None:
        encoder = backbone.vision_encoder
        shape = (
            -1,
            encoder.in_channels,
            encoder.temporal_patch_size,
            encoder.patch_size,
            encoder.patch_size,
        )
        pixels = pixels.numpy().reshape(shape).transpose(0, 2, 3, 4, 1)
        inputs["pixel_values"] = pixels[None]
        inputs["image_grid_thw"] = grid.numpy().astype("int32")[None]
        inputs["vision_indices"] = vision_positions[None]
    return inputs, seq_len


def fill_missing_vision_inputs(model, inputs):
    # Text-only runs still need every functional input to be fed.
    for tensor in model.inputs:
        name = tensor.name
        if name not in inputs:
            shape = [0 if d is None else d for d in tensor.shape]
            inputs[name] = np.zeros(shape, dtype=tensor.dtype)
    return inputs


def keras_forward(backbone, kind, hf_inputs, position_ids, vision_positions):
    inputs, _ = keras_inputs(backbone, kind, hf_inputs, vision_positions)
    # The backbone takes (batch, seq_len, 3) positions.
    inputs["position_ids"] = np.transpose(position_ids, (0, 2, 1))
    layers = [
        backbone.get_layer(f"transformer_layer_{i}").output
        for i in range(backbone.num_layers)
    ]
    probe = keras.Model(backbone.inputs, layers + [backbone.output])
    inputs = fill_missing_vision_inputs(probe, inputs)
    outputs = probe(inputs)
    result = {
        f"layer_{i}": keras.ops.convert_to_numpy(o)
        for i, o in enumerate(outputs[:-1])
    }
    hidden = outputs[-1]
    logits = backbone.get_layer("token_embedding")(hidden, reverse=True)
    result["logits"] = keras.ops.convert_to_numpy(logits)
    if "pixel_values" in inputs and inputs["pixel_values"].shape[1] > 0:
        features, deepstack = backbone.vision_encoder(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )
        result["vision"] = keras.ops.convert_to_numpy(features)
        for i, feat in enumerate(deepstack):
            result[f"deepstack_{i}"] = keras.ops.convert_to_numpy(feat)
    return result


def compare(name, expected, actual, atol):
    actual = np.asarray(actual).reshape(expected.shape)
    diff = np.abs(expected.astype("float64") - actual.astype("float64"))
    within = 100.0 * float((diff <= atol).mean())
    print(
        f"  {name:<14} max abs diff {diff.max():.3e}   within {atol}: "
        f"{within:6.2f}%"
    )
    return float(diff.max())


def run(tie_word_embeddings, atol, seed):
    print(f"\n=== tie_word_embeddings={tie_word_embeddings} ===")
    torch.manual_seed(seed)
    config = build_config(tie_word_embeddings)
    hf_model = Qwen3VLForConditionalGeneration(config)
    randomize_(hf_model, seed)
    worst = 0.0
    with tempfile.TemporaryDirectory() as path:
        hf_model.save_pretrained(path)
        hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, attn_implementation="eager", dtype=torch.float32
        ).eval()
        backbone = keras_hub.models.Qwen3_VLBackbone.from_preset(
            path, dtype="float32"
        )
        hf_params = sum(p.numel() for p in hf_model.parameters())
        keras_params = backbone.count_params()
        print(f"  params: HF {hf_params:,}  KerasHub {keras_params:,}")
        if hf_params != keras_params:
            worst = float("inf")

        for kind in ("text", "image", "video"):
            print(f"\n[{kind}]")
            hf_inputs, vision_positions = make_inputs(kind, seed)
            reference = hf_reference(hf_model, kind, hf_inputs)
            actual = keras_forward(
                backbone,
                kind,
                hf_inputs,
                reference["position_ids"],
                vision_positions,
            )
            for name, expected in reference.items():
                if name == "position_ids":
                    continue
                if name not in actual:
                    print(f"  {name:<14} MISSING on KerasHub side")
                    worst = float("inf")
                    continue
                worst = max(worst, compare(name, expected, actual[name], atol))
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    worst = max(run(tie, args.atol, args.seed) for tie in (True, False))
    print(f"\nWorst max abs diff: {worst:.3e}")
    if worst > args.atol:
        print("FAILED")
        sys.exit(1)
    print("PASSED")


if __name__ == "__main__":
    main()
