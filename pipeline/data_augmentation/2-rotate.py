#!/usr/bin/env python3
"""Rotate images by 90 degrees counter-clockwise."""
import tensorflow as tf


def rotate_image(image):
    """Rotate an image 90 degrees counter-clockwise.

    Args:
        image: A `tf.Tensor` representing the image to rotate.

    Returns:
        A `tf.Tensor` containing the rotated image.
    """
    return tf.image.rot90(image, k=1)
