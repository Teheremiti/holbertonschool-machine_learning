#!/usr/bin/env python3
""" create _layer function """
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for the input data.

    Args:
        prev (Tensor): The tensor output of the previous layer.
        n (int): The number of nodes in the layer to create.
        activation (str): The activation function that the layer should use.

    Returns:
        The tensor output of the layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_initializer=initializer, name="layer")
    return layer(prev)
