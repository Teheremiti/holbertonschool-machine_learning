#!/usr/bin/env python3
"""Adjust image hue."""
import tensorflow as tf


def change_hue(image, delta):
    """Change an image's hue.

    Args:
        image: A `tf.Tensor` representing the image.
        delta: Amount to shift the hue.

    Returns:
        A `tf.Tensor` containing the hue-adjusted image.
    """
    return tf.image.adjust_hue(image, delta)
