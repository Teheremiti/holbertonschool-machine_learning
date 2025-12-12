#!/usr/bin/env python3
"""Dataset class for Portuguese to English machine translation."""
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Loads and prepares a dataset for machine translation.

    Attributes:
        data_train: tf.data.Dataset containing training sentence pairs.
        data_valid: tf.data.Dataset containing validation sentence pairs.
        tokenizer_pt: Tokenizer for Portuguese.
        tokenizer_en: Tokenizer for English.
    """

    def __init__(self):
        """Initializes the Dataset by loading data and creating tokenizers."""
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for the dataset.

        Args:
            data: tf.data.Dataset containing tuples of (pt, en) where:
                - pt: tf.Tensor containing the Portuguese sentence.
                - en: tf.Tensor containing the English sentence.

        Returns:
            tuple: (tokenizer_pt, tokenizer_en)
                - tokenizer_pt: Portuguese tokenizer.
                - tokenizer_en: English tokenizer.
        """
        def pt_iterator():
            """Yields Portuguese sentences from the dataset."""
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_iterator():
            """Yields English sentences from the dataset."""
            for _, en in data:
                yield en.numpy().decode('utf-8')

        # Load pretrained tokenizers as base
        base_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        base_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        # Train new tokenizers on the training data
        vocab_size = 2 ** 13
        tokenizer_pt = base_pt.train_new_from_iterator(
            pt_iterator(), vocab_size=vocab_size
        )
        tokenizer_en = base_en.train_new_from_iterator(
            en_iterator(), vocab_size=vocab_size
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encodes a translation pair into tokens.

        Args:
            pt: tf.Tensor containing the Portuguese sentence.
            en: tf.Tensor containing the corresponding English sentence.

        Returns:
            tuple: (pt_tokens, en_tokens)
                - pt_tokens: np.ndarray containing the Portuguese tokens.
                - en_tokens: np.ndarray containing the English tokens.
        """
        # Decode tensors to strings
        pt_str = pt.numpy().decode('utf-8')
        en_str = en.numpy().decode('utf-8')

        # Tokenize without special tokens (we add our own)
        pt_tokens = self.tokenizer_pt.encode(pt_str, add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_str, add_special_tokens=False)

        # Get vocab sizes for start/end tokens
        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        # Add start (vocab_size) and end (vocab_size + 1) tokens
        pt_tokens = [vocab_size_pt] + pt_tokens + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + en_tokens + [vocab_size_en + 1]

        # Convert to numpy arrays
        pt_tokens = tf.constant(pt_tokens).numpy()
        en_tokens = tf.constant(en_tokens).numpy()

        return pt_tokens, en_tokens
