#!/usr/bin/env python3
"""Shear images using Keras image preprocessing."""
from tensorflow import keras as k


def shear_image(image, intensity):
    """Randomly shear an image.

    This function converts the input tensor to a NumPy array before applying
    the Keras shear transform.

    Args:
        image: A `tf.Tensor` to shear (eager tensor).
        intensity: Float controlling the amount of shearing.

    Returns:
        A NumPy array containing the sheared image.
    """
    image_np = image.numpy()
    return k.preprocessing.image.random_shear(image_np,
                                              intensity,
                                              row_axis=0,
                                              col_axis=1,
                                              channel_axis=2)
