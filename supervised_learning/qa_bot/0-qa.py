#!/usr/bin/env python3
"""Simple single-reference question answering using BERT."""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


# Lazily-initialized global tokenizer and model reused across calls.
_QA_TOKENIZER = None
_QA_MODEL = None


def _get_tokenizer():
    """Return a cached BERT tokenizer instance."""
    global _QA_TOKENIZER
    if _QA_TOKENIZER is None:
        _QA_TOKENIZER = BertTokenizer.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad"
        )
    return _QA_TOKENIZER


def _get_model():
    """Return a cached BERT question-answering model."""
    global _QA_MODEL
    if _QA_MODEL is None:
        _QA_MODEL = hub.load(
            "https://www.kaggle.com/models/seesee/bert/"
            "TensorFlow2/uncased-tf2-qa/1"
        )
    return _QA_MODEL


def question_answer(question, reference):
    """Find an answer span within a reference document.

    The function uses a fine-tuned BERT model to select the most probable
    start and end token indices for the answer span inside the reference
    text.

    Args:
        question (str): Question to answer.
        reference (str): Reference document from which to find the answer.

    Returns:
        str or None: Extracted answer text. If the model does not predict
        any span (no tokens in the range), None is returned.
    """

    tokenizer = _get_tokenizer()
    model = _get_model()

    # Tokenize inputs.
    q_tokenized = tokenizer.tokenize(question)
    ref_tokenized = tokenizer.tokenize(reference)

    # Concatenate tokens with special tokens.
    tokens = ["[CLS]"] + q_tokenized + ["[SEP]"] + ref_tokenized + ["[SEP]"]

    # Convert tokens to ids.
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Build input masks and segment ids.
    input_mask = [1] * len(input_word_ids)
    input_type_ids = (
        [0] * (1 + len(q_tokenized) + 1) + [1] * (len(ref_tokenized) + 1)
    )

    # Convert to TensorFlow tensors and add batch dimension.
    input_word_ids, input_mask, input_type_ids = map(
        lambda t: tf.expand_dims(tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids),
    )

    # Run the model and get start and end logits.
    outputs = model([input_word_ids, input_mask, input_type_ids])

    # Find the most probable start and end tokens (+1 to ignore [CLS]).
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1

    # Extract tokens corresponding to the predicted answer span.
    answer_tokens = tokens[short_start: short_end + 1]
    if not answer_tokens:
        return None

    # Convert tokens back to a human-readable string.
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer
