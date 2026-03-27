#!/usr/bin/env python3
"""Flip images horizontally."""
import tensorflow as tf


def flip_image(image):
    """Flip an image horizontally.

    Args:
        image: A `tf.Tensor` representing the image to flip.

    Returns:
        A `tf.Tensor` containing the flipped image.
    """
    return tf.image.flip_left_right(image)
