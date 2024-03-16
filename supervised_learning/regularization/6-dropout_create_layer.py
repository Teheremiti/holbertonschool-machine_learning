#!/usr/bin/env python3
""" Create a layer with Dropout """
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout.

    Args:
        prev (tensor): The output of the previous layer.
        n (int): The number of nodes the new layer should contain.
        activation (str): The activation function that should be used on the
            layer.
        keep_prob (float): The probability that a node will be kept.

    Returns:
        The output of the new layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode='fan_avg')
    dropout_layer = tf.compat.v1.layers.Dropout(rate=keep_prob)
    new_layer = tf.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=dropout_layer,
        name="layer"
    )
    output = new_layer(prev)
    return output
