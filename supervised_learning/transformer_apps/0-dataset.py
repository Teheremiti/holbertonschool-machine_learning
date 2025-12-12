#!/usr/bin/env python3
"""Dataset class for Portuguese to English machine translation.

This module provides the Dataset class that loads and preprocesses the
TED Talks Portuguese-English translation dataset for transformer training.
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Loads and prepares a dataset for machine translation.

    This class handles loading the TED HRLR Portuguese to English translation
    dataset and creating subword tokenizers for both languages.

    Attributes:
        data_train: tf.data.Dataset containing training sentence pairs.
        data_valid: tf.data.Dataset containing validation sentence pairs.
        tokenizer_pt: SubwordTextEncoder for Portuguese tokenization.
        tokenizer_en: SubwordTextEncoder for English tokenization.
    """

    def __init__(self):
        """Initializes the Dataset by loading data and creating tokenizers.

        Loads the TED HRLR Portuguese to English translation dataset and
        builds subword tokenizers for both languages from the training data.
        """
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

        Builds SubwordTextEncoder tokenizers for both Portuguese and English
        languages using the provided dataset corpus.

        Args:
            data: tf.data.Dataset containing tuples of (pt, en) where:
                - pt: tf.Tensor containing the Portuguese sentence.
                - en: tf.Tensor containing the English sentence.

        Returns:
            tuple: A tuple containing:
                - tokenizer_pt: SubwordTextEncoder for Portuguese with
                    target vocabulary size of 2^15.
                - tokenizer_en: SubwordTextEncoder for English with
                    target vocabulary size of 2^15.
        """
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data), target_vocab_size=2 ** 15)
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data), target_vocab_size=2 ** 15)
        return tokenizer_pt, tokenizer_en
