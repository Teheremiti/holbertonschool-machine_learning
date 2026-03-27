#!/usr/bin/env python3
"""Randomly adjust image contrast."""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly adjust the contrast of an image.

    Args:
        image: A 3D ``tf.Tensor`` representing the input image.
        lower: Float lower bound for the random contrast factor.
        upper: Float upper bound for the random contrast factor.

    Returns:
        The contrast-adjusted image tensor.
    """
    return tf.image.random_contrast(image, lower, upper)
