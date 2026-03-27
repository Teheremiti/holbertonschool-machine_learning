#!/usr/bin/env python3
"""Crop images randomly."""
import tensorflow as tf


def crop_image(image, size):
    """Randomly crop an image.

    Args:
        image: A `tf.Tensor` representing the image to crop.
        size: The output crop size (e.g., `[height, width, channels]`).

    Returns:
        A `tf.Tensor` containing the cropped image.
    """
    return tf.image.random_crop(image, size)
