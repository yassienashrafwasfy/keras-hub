import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.mobileclip2.mobileclip2_backbone import (
    MobileCLIP2Backbone,
)
from keras_hub.src.models.mobileclip2.mobileclip2_image_converter import (
    MobileCLIP2ImageConverter,
)
from keras_hub.src.models.mobileclip2.mobileclip2_tokenizer import (
    MobileCLIP2Tokenizer,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.MobileCLIP2Preprocessor")
class MobileCLIP2Preprocessor(CausalLMPreprocessor):
    """MobileCLIP2 preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.MobileCLIP2Backbone`. By default, it will take in batches
    of strings and images, and return token ids and resized images.

    Args:
        tokenizer: A `keras_hub.models.MobileCLIP2Tokenizer` instance.
        image_converter: A `keras_hub.models.MobileCLIP2ImageConverter`
            instance.
        sequence_length: The length of the packed inputs.
        add_start_token: If `True`, the preprocessor will prepend the tokenizer
            start token to each input sequence.
        add_end_token: If `True`, the preprocessor will append the tokenizer
            end token to each input sequence.
        to_lower: bool. Whether to lower the inputs.

    Call arguments:
        x: A dict with `"prompts"` and `"images"` keys, where `"prompts"` is
            `tf.Tensor` or list of python strings and `"images"` are the image
            tensors.
        y: Label data. Should always be `None` since MobileCLIP2 doesn't need
            the label to calculate the loss.
        sample_weight: Label weights.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.

    Examples:
    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_hub.models.MobileCLIP2Preprocessor.from_preset(
        "mobileclip2_b"
    )

    # Tokenize the sentence and preprocess the image.
    preprocessor(
        {
            "prompts": "The quick brown fox jumped.",
            "images": np.ones(shape=(123, 123, 3)),
        }
    )

    # Tokenize a batch of sentences and preprocess a batch of images.
    preprocessor(
        {
            "prompts": ["The quick brown fox jumped.", "The fox slept."],
            "images": np.ones(shape=(2, 123, 123, 3)),
        }
    )
    ```
    """

    backbone_cls = MobileCLIP2Backbone
    tokenizer_cls = MobileCLIP2Tokenizer
    image_converter_cls = MobileCLIP2ImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=77,
        add_start_token=True,
        add_end_token=True,
        to_lower=True,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            add_start_token=add_start_token,
            add_end_token=add_end_token,
            **kwargs,
        )
        self.image_converter = image_converter
        self.to_lower = to_lower

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        self.packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.sequence_length,
            return_padding_mask=True,
        )
        self.built = True

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length
        images, prompts = x["images"], x["prompts"]
        if self.to_lower:
            prompts = tf.strings.lower(prompts)
        prompts = self.tokenizer(prompts)
        if images is not None and self.image_converter:
            images = self.image_converter(images)
        token_ids, padding_mask = self.packer(
            prompts,
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        x = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "images": images,
        }
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "to_lower": self.to_lower,
            }
        )
        return config
