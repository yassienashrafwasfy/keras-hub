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
ATOL = 1e-3

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


def hf_text_reference(hf_model, hf_tokenizer):
    ids = hf_tokenizer(TEXT_PROMPT, return_tensors="np")["input_ids"]
    with torch.no_grad():
        logits = hf_model(input_ids=torch.tensor(ids)).logits
    return {"input_ids": ids, "logits": logits.float().numpy()}


def hf_media_reference(hf_model, processor, **processor_kwargs):
    inputs = processor(return_tensors="pt", **processor_kwargs)
    with torch.no_grad():
        logits = hf_model(**inputs).logits
    return {
        "input_ids": inputs["input_ids"].numpy(),
        "logits": logits.float().numpy(),
    }


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
            fps=VIDEO_FPS,
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
    diff = np.abs(logits - hf_logits)
    print(
        f"  {name}: logit max abs diff {diff.max():.6f}, mean {diff.mean():.6f}"
    )
    return float(diff.max())


def check_token_ids(preprocessed, hf_ids):
    mask = ops.convert_to_numpy(preprocessed["padding_mask"]).astype(bool)
    ids = ops.convert_to_numpy(preprocessed["token_ids"])[mask]
    np.testing.assert_array_equal(ids, hf_ids[0])


def verify(keras_model, hf_results, hf_param_count):
    print("\nPARAMETER COUNT")
    keras_params = count_unique_params(keras_model.backbone)
    print(f"  KerasHub {keras_params:,}  HF {hf_param_count:,}")
    assert keras_params == hf_param_count, "Parameter counts differ."

    print("\nLOGITS")
    preprocessor = keras_model.preprocessor
    worst = {}

    ref = hf_results["text"]
    pre = preprocessor.generate_preprocess(
        [TEXT_PROMPT], sequence_length=ref["input_ids"].shape[1]
    )
    check_token_ids(pre, ref["input_ids"])
    worst["text"] = compare_logits("text", keras_model, pre, ref["logits"])

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

    ref = hf_results["video"]
    pre = preprocessor.generate_preprocess(
        {
            "prompts": [
                "<|im_start|>user\n<|vision_start|><|video_pad|>"
                "<|vision_end|>Describe this video.<|im_end|>\n"
                "<|im_start|>assistant\n"
            ],
            "videos": [hf_results["raw_frames"]],
        },
        sequence_length=ref["input_ids"].shape[1],
    )
    check_token_ids(pre, ref["input_ids"])
    worst["video"] = compare_logits("video", keras_model, pre, ref["logits"])

    for name, value in worst.items():
        status = "ok" if value <= ATOL else "ABOVE TOLERANCE"
        print(f"  {name}: {status} (atol={ATOL})")

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
