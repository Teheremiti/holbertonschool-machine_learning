#!/usr/bin/env python3
"""Mask creation utilities for transformer training."""
import tensorflow as tf


def create_masks(inputs, target):
    """Creates all masks required for transformer training and validation.

    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in) containing
            the input sentence.
        target: tf.Tensor of shape (batch_size, seq_len_out) containing
            the target sentence.

    Returns:
        tuple: (encoder_mask, combined_mask, decoder_mask)
            - encoder_mask: tf.Tensor of shape (batch_size, 1, 1, seq_len_in),
                padding mask applied in the encoder.
            - combined_mask: tf.Tensor of shape
                (batch_size, 1, seq_len_out, seq_len_out), combined padding
                and look-ahead mask for the 1st decoder attention block.
            - decoder_mask: tf.Tensor of shape (batch_size, 1, 1, seq_len_in),
                padding mask for the 2nd decoder attention block.
    """
    # Encoder padding mask - masks padding in input for encoder self-attention
    encoder_mask = create_padding_mask(inputs)

    # Decoder padding mask - masks input padding for encoder-decoder attention
    decoder_mask = create_padding_mask(inputs)

    # Look-ahead mask to prevent attending to future tokens
    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = create_look_ahead_mask(seq_len_out)

    # Target padding mask
    target_padding_mask = create_padding_mask(target)

    # Combined mask is maximum of look-ahead and target padding masks
    combined_mask = tf.maximum(look_ahead_mask, target_padding_mask)

    return encoder_mask, combined_mask, decoder_mask


def create_padding_mask(seq):
    """Creates a padding mask for a sequence.

    Args:
        seq: tf.Tensor containing sequence token IDs.

    Returns:
        tf.Tensor: Padding mask of shape (batch_size, 1, 1, seq_len)
            where padding positions (0) are 1.0 and valid positions are 0.0.
    """
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """Creates a look-ahead mask to prevent attending to future tokens.

    Args:
        size: The sequence length for the square mask.

    Returns:
        tf.Tensor: Look-ahead mask of shape (1, 1, size, size) where
            future positions are 1.0 and valid positions are 0.0.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask[tf.newaxis, tf.newaxis, :, :]
