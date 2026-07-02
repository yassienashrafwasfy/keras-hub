from keras_hub.src.models.mobileclip2.mobileclip2_tokenizer import (
    MobileCLIP2Tokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class MobileCLIP2TokenizerTest(TestCase):
    def setUp(self):
        merges = ["a i", "p l", "n e</w>", "p o", "r t</w>", "ai r", "pl a"]
        merges += ["po rt</w>", "pla ne</w>"]
        self.merges = merges
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["<|endoftext|>", "<|startoftext|>"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.init_kwargs = {"vocabulary": self.vocab, "merges": self.merges}
        self.input_data = ["airplane ", " airport"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=MobileCLIP2Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            # Whitespaces should be removed.
            expected_output=[[4, 14], [4, 16]],
            expected_detokenize_output=["airplane", "airport"],
        )

    def test_pad_with_end_token(self):
        init_kwargs = self.init_kwargs.copy()
        init_kwargs["pad_with_end_token"] = True
        tokenizer = MobileCLIP2Tokenizer(**init_kwargs)
        self.assertEqual(tokenizer.pad_token_id, tokenizer.end_token_id)

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            MobileCLIP2Tokenizer(
                vocabulary={"foo": 0, "bar": 1}, merges=["fo o"]
            )
