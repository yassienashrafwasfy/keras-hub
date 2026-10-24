"""Convert Qwen3-VL HuggingFace checkpoints to KerasHub presets.

The preset is saved before any verification runs, and verification uses a
fresh float32 reload of the saved preset, so nothing done while checking
numerics can leak into the files on disk.

Usage:
    python tools/checkpoint_conversion/convert_qwen3_vl_checkpoints.py \
        --preset qwen3_vl_2b_instruct
"""

import gc
import os
import random
import tempfile
from io import BytesIO

os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import keras
import numpy as np
import requests
import torch
from absl import app
from absl import flags
from keras import ops
from PIL import Image
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
from transformers import AutoTokenizer
from transformers.video_utils import VideoMetadata

import keras_hub

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.set_default_device(torch.device("cpu"))

PRESET_MAP = {
    "qwen3_vl_2b_instruct": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3_vl_2b_thinking": "Qwen/Qwen3-VL-2B-Thinking",
    "qwen3_vl_4b_instruct": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3_vl_4b_thinking": "Qwen/Qwen3-VL-4B-Thinking",
    "qwen3_vl_8b_instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3_vl_8b_thinking": "Qwen/Qwen3-VL-8B-Thinking",
    "qwen3_vl_32b_instruct": "Qwen/Qwen3-VL-32B-Instruct",
    "qwen3_vl_32b_thinking": "Qwen/Qwen3-VL-32B-Thinking",
}

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
VIDEO_URL = (
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/"
    "Big_Buck_Bunny_360_10s_1MB.mp4"
)
VIDEO_FPS = 2
MAX_NEW_TOKENS = 32
MAX_VIDEO_FRAMES = 8
TEXT_PROMPT = "What is Keras?"
IMAGE_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe this image.<|im_end|>\n<|im_start|>assistant\n"
)
# Relative to the largest reference value, so it holds for any logit scale.
RTOL = 1e-3

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)
flags.DEFINE_string(
    "output_dir", None, "Where to save the preset. Defaults to ./<preset>."
)
flags.DEFINE_bool(
    "skip_generation",
    False,
    "Only compare logits, skip text generation.",
)


def count_unique_params(model):
    unique = {id(w): w for w in model.weights}.values()
    return sum(int(np.prod(w.shape)) for w in unique)


def load_video_frames():
    """Download a clip and sample frames at VIDEO_FPS as PIL images."""
    import av

    response = requests.get(VIDEO_URL, timeout=60)
    response.raise_for_status()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "clip.mp4")
        with open(path, "wb") as f:
            f.write(response.content)
        with av.open(path) as container:
            stream = container.streams.video[0]
            source_fps = float(stream.average_rate)
            video = np.stack(
                [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
            )
    total = video.shape[0]
    count = max(4, min(int(total / source_fps * VIDEO_FPS), MAX_VIDEO_FRAMES))
    indices = np.linspace(0, total - 1, count).round().astype(int)
    return [Image.fromarray(video[i]) for i in indices]


def hf_forward(hf_model, **inputs):
    """Run HF once and keep the logits and every intermediate output."""
    layer_outputs = []
    hooks = [
        layer.register_forward_hook(
            lambda module, args, output: layer_outputs.append(
                (output[0] if isinstance(output, tuple) else output)
                .float()
                .numpy()
            )
        )
        for layer in hf_model.model.language_model.layers
    ]
    with torch.no_grad():
        logits = hf_model(**inputs).logits
        components = {}
        pixel_values = None
        for pixels, grid in (
            ("pixel_values", "image_grid_thw"),
            ("pixel_values_videos", "video_grid_thw"),
        ):
            if pixels in inputs:
                pixel_values = inputs[pixels].float().numpy()
                vision = hf_model.model.visual(
                    inputs[pixels], grid_thw=inputs[grid]
                )
                components["vision"] = vision.pooler_output.float().numpy()
                for i, feature in enumerate(vision.deepstack_features):
                    components[f"deepstack_{i}"] = feature.float().numpy()
    for hook in hooks:
        hook.remove()
    for i, output in enumerate(layer_outputs):
        components[f"layer_{i}"] = output
    return {
        "input_ids": inputs["input_ids"].numpy(),
        "logits": logits.float().numpy(),
        "components": components,
        "pixel_values": pixel_values,
    }


def hf_text_reference(hf_model, hf_tokenizer):
    ids = hf_tokenizer(TEXT_PROMPT, return_tensors="pt")["input_ids"]
    return hf_forward(hf_model, input_ids=ids)


def hf_media_reference(hf_model, processor, **processor_kwargs):
    inputs = processor(return_tensors="pt", **processor_kwargs)
    return hf_forward(hf_model, **inputs)


def precompute_hf_outputs(hf_model, hf_tokenizer, hf_preset):
    """Run every HF forward pass so the HF model can be freed afterwards."""
    processor = AutoProcessor.from_pretrained(hf_preset)
    image = Image.open(
        BytesIO(requests.get(IMAGE_URL, timeout=30).content)
    ).convert("RGB")
    frames = load_video_frames()
    results = {
        "text": hf_text_reference(hf_model, hf_tokenizer),
        "image": hf_media_reference(
            hf_model, processor, text=[IMAGE_PROMPT], images=[image]
        ),
        "video": hf_media_reference(
            hf_model,
            processor,
            text=[
                "<|im_start|>user\n<|vision_start|><|video_pad|>"
                "<|vision_end|>Describe this video.<|im_end|>\n"
                "<|im_start|>assistant\n"
            ],
            videos=[frames],
            video_metadata=[
                VideoMetadata(
                    total_num_frames=len(frames),
                    fps=VIDEO_FPS,
                    duration=len(frames) / VIDEO_FPS,
                    frames_indices=list(range(len(frames))),
                )
            ],
            do_sample_frames=False,
        ),
        "raw_image": image,
        "raw_frames": np.stack([np.array(f) for f in frames]),
    }
    if not FLAGS.skip_generation:
        with torch.no_grad():
            out = hf_model.generate(
                input_ids=torch.tensor(results["text"]["input_ids"]),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        results["text"]["generated"] = hf_tokenizer.decode(
            out[0], skip_special_tokens=True
        )
    return results


def parity(value, expected):
    """Return max abs, mean abs, relative max diff and cosine of two arrays."""
    value = value.reshape(expected.shape).astype(np.float64)
    expected = expected.astype(np.float64)
    diff = np.abs(value - expected)
    cosine = (value.ravel() @ expected.ravel()) / (
        np.linalg.norm(value) * np.linalg.norm(expected) + 1e-30
    )
    rel = diff.max() / max(np.abs(expected).max(), 1e-12)
    return diff.max(), diff.mean(), rel, cosine


ROW_HEADER = (
    f"    {'component':20s} {'max abs':>10s} {'mean abs':>10s} "
    f"{'rel max':>10s} {'cosine':>10s}"
)


def print_row(name, value, expected):
    max_abs, mean_abs, rel, cosine = parity(value, expected)
    flag = "  HIGH" if rel > RTOL else ""
    print(
        f"    {name:20s} {max_abs:10.2e} {mean_abs:10.2e} "
        f"{rel:10.2e} {cosine:10.6f}{flag}"
    )
    return rel


def compare_logits(name, keras_model, preprocessed, hf_logits):
    backbone = keras_model.backbone
    inputs = {
        k: ops.convert_to_tensor(v)
        for k, v in preprocessed.items()
        if k in backbone.input
    }
    hidden = backbone(inputs)
    logits = backbone.token_embedding(hidden, reverse=True)
    logits = ops.convert_to_numpy(logits).astype(np.float32)
    max_abs, mean_abs, rel, _ = parity(logits, hf_logits)
    print(
        f"  {name}: logit max abs diff {max_abs:.6f}, mean {mean_abs:.6f}, "
        f"relative max {rel:.2e}"
    )
    return rel


def compare_components(name, keras_model, preprocessed, hf_components):
    """Print the parity of the vision tower and of every decoder layer."""
    backbone = keras_model.backbone
    feed = {
        k: ops.convert_to_tensor(v)
        for k, v in preprocessed.items()
        if k in backbone.input
    }
    layers = [
        backbone.get_layer(f"transformer_layer_{i}").output
        for i in range(backbone.num_layers)
    ]
    probe = keras.Model(backbone.inputs, layers)
    for tensor in probe.inputs:  # a text prompt leaves the vision inputs out
        if tensor.name not in feed:
            shape = [0 if d is None else d for d in tensor.shape]
            feed[tensor.name] = ops.zeros(shape, dtype=tensor.dtype)
    got = {
        f"layer_{i}": ops.convert_to_numpy(output).astype(np.float32)
        for i, output in enumerate(probe(feed))
    }
    if feed["pixel_values"].shape[1] > 0:
        features, *deepstack = backbone.vision_encoder(
            {
                "pixel_values": feed["pixel_values"],
                "image_grid_thw": feed["image_grid_thw"],
            }
        )
        got["vision"] = ops.convert_to_numpy(features).astype(np.float32)
        for i, feature in enumerate(deepstack):
            got[f"deepstack_{i}"] = ops.convert_to_numpy(feature).astype(
                np.float32
            )

    print(f"  {name} components (flag above relative {RTOL}):")
    print(ROW_HEADER)
    for component, expected in hf_components.items():
        print_row(component, got[component], expected)


def compare_vision_inputs(name, keras_model, preprocessed, hf_reference):
    """Separate the preprocessing from the vision tower.

    Compares the pixels both preprocessors produce, then feeds the HF
    pixels to the KerasHub vision tower.
    """
    vision_encoder = keras_model.backbone.vision_encoder
    # HF packs (patches, C * T * P * P), channel first. KerasHub wants
    # (patches, T, P, P, C).
    hf_as_keras = (
        hf_reference["pixel_values"]
        .reshape(
            -1,
            vision_encoder.in_channels,
            vision_encoder.temporal_patch_size,
            vision_encoder.patch_size,
            vision_encoder.patch_size,
        )
        .transpose(0, 2, 3, 4, 1)
    )
    print(f"  {name} vision inputs:")
    print(ROW_HEADER)
    print_row(
        "pixel_values",
        ops.convert_to_numpy(preprocessed["pixel_values"]),
        hf_as_keras,
    )
    features, *deepstack = vision_encoder(
        {
            "pixel_values": ops.convert_to_tensor(hf_as_keras[None]),
            "image_grid_thw": ops.convert_to_tensor(
                preprocessed["image_grid_thw"]
            ),
        }
    )
    for i, value in enumerate([features, *deepstack]):
        component = "vision" if i == 0 else f"deepstack_{i - 1}"
        print_row(
            f"{component} (HF pixels)",
            ops.convert_to_numpy(value),
            hf_reference["components"][component],
        )


def check_token_ids(preprocessed, hf_ids):
    mask = ops.convert_to_numpy(preprocessed["padding_mask"]).astype(bool)
    ids = ops.convert_to_numpy(preprocessed["token_ids"])[mask]
    np.testing.assert_array_equal(ids, hf_ids[0])


def verify(keras_model, hf_results, hf_param_count):
    print("\nPARAMETER COUNT")
    keras_params = count_unique_params(keras_model.backbone)
    print(f"  KerasHub {keras_params:,}  HF {hf_param_count:,}")
    assert keras_params == hf_param_count, "Parameter counts differ."

    print("\nLOGITS AND COMPONENTS")
    preprocessor = keras_model.preprocessor
    worst = {}

    ref = hf_results["text"]
    pre = preprocessor.generate_preprocess(
        [TEXT_PROMPT], sequence_length=ref["input_ids"].shape[1]
    )
    check_token_ids(pre, ref["input_ids"])
    worst["text"] = compare_logits("text", keras_model, pre, ref["logits"])
    compare_components("text", keras_model, pre, ref["components"])

    ref = hf_results["image"]
    pre = preprocessor.generate_preprocess(
        {
            "prompts": [IMAGE_PROMPT],
            "images": [np.array(hf_results["raw_image"])],
        },
        sequence_length=ref["input_ids"].shape[1],
    )
    check_token_ids(pre, ref["input_ids"])
    worst["image"] = compare_logits("image", keras_model, pre, ref["logits"])
    compare_components("image", keras_model, pre, ref["components"])
    compare_vision_inputs("image", keras_model, pre, ref)

    ref = hf_results["video"]
    pre = preprocessor.generate_preprocess(
        {
            "prompts": [
                "<|im_start|>user\n<|vision_start|><|video_pad|>"
                "<|vision_end|>Describe this video.<|im_end|>\n"
                "<|im_start|>assistant\n"
            ],
            "videos": [hf_results["raw_frames"]],
            "video_fps": [[VIDEO_FPS]],
        },
        sequence_length=ref["input_ids"].shape[1],
    )
    check_token_ids(pre, ref["input_ids"])
    worst["video"] = compare_logits("video", keras_model, pre, ref["logits"])
    compare_components("video", keras_model, pre, ref["components"])
    compare_vision_inputs("video", keras_model, pre, ref)

    for name, value in worst.items():
        status = "ok" if value <= RTOL else "ABOVE TOLERANCE"
        print(f"  {name}: {status} (relative max diff, rtol={RTOL})")

    if not FLAGS.skip_generation:
        print("\nTEXT GENERATION")
        print(f"  HF:       {hf_results['text']['generated']}")
        keras_model.compile(sampler="greedy")
        prompt_length = hf_results["text"]["input_ids"].shape[1]
        generated = keras_model.generate(
            TEXT_PROMPT, max_length=prompt_length + MAX_NEW_TOKENS
        )
        print(f"  KerasHub: {generated}")
        print(
            "  match:   ",
            generated.strip() == hf_results["text"]["generated"].strip(),
        )


def main(_):
    preset = FLAGS.preset
    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset '{preset}'. Must be one of "
            f"{', '.join(PRESET_MAP.keys())}"
        )
    hf_preset = PRESET_MAP[preset]
    output_dir = FLAGS.output_dir or f"./{preset}"

    print("-> Loading HF model...")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        hf_preset, device_map="cpu", dtype=torch.float32
    ).eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
    hf_param_count = sum(p.numel() for p in hf_model.parameters())
    print(f"   {hf_param_count:,} params")

    print("-> Precomputing HF outputs...")
    hf_results = precompute_hf_outputs(hf_model, hf_tokenizer, hf_preset)
    del hf_model, hf_tokenizer
    gc.collect()

    # Save first, in the checkpoint dtype, before anything touches the model.
    print(f"-> Converting and saving preset to {output_dir}...")
    keras_hub.models.Qwen3_VLCausalLM.from_preset(
        f"hf://{hf_preset}", dtype="bfloat16"
    ).save_to_preset(output_dir)
    gc.collect()

    print("-> Verifying the saved preset in float32...")
    keras_model = keras_hub.models.Qwen3_VLCausalLM.from_preset(
        output_dir, dtype="float32"
    )
    verify(keras_model, hf_results, hf_param_count)
    print("\nDone.")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
