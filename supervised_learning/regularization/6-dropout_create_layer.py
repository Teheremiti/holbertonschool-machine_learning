#!/usr/bin/env python3
""" Create a layer with Dropout """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Args:
        prev (tensor): The output of the previous layer.
        n (int): The number of nodes the new layer should contain.
        activation (str): The activation function that should be used on the
            layer.
        keep_prob (float): The probability that a node will be kept.
        training (bool): Defines whether the model is in training mode or not.

    Returns:
        The output of the new layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg')

    dense_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )

    dropout_layer = tf.keras.layers.Dropout(rate=1 - keep_prob)
    dropout_output = dropout_layer(dense_layer(prev), training=training)

    return dropout_output
