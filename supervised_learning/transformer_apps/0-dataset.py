#!/usr/bin/env python3
"""Dataset class for Portuguese to English machine translation."""
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
