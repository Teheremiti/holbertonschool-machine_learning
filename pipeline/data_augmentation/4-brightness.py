#!/usr/bin/env python3
"""Adjust image brightness."""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Change an image's brightness.

    Args:
        image: A `tf.Tensor` representing the image.
        max_delta: Brightness delta applied to the image.

    Returns:
        A `tf.Tensor` containing the brightness-adjusted image.
    """
    return tf.image.random_brightness(image, max_delta)
