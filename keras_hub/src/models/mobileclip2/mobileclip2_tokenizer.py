from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.clip.clip_tokenizer import CLIPTokenizer
from keras_hub.src.models.mobileclip2.mobileclip2_backbone import (
    MobileCLIP2Backbone,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.MobileCLIP2Tokenizer",
        "keras_hub.models.MobileCLIP2Tokenizer",
    ]
)
class MobileCLIP2Tokenizer(CLIPTokenizer):
    """A MobileCLIP2 tokenizer using Byte-Pair Encoding subword segmentation.

    MobileCLIP2 uses the same 49408-token CLIP BPE vocabulary and special
    tokens as `keras_hub.models.CLIPTokenizer`; this subclass only rebinds
    `backbone_cls` so that presets resolve to `MobileCLIP2Backbone`.

    Args:
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules. The merge rule file
            should have one merge rule per line. Every merge rule contains
            merge entities separated by a space.
        pad_with_end_token: bool. Whether to pad the output with `end_token`.
    """

    backbone_cls = MobileCLIP2Backbone
