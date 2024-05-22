#!/usr/bin/env python3
""" L2 regularization - Create layer """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes L2 regularization.

    Args:
        prev (tensor): The output of the previous layer.
        n (int): The number of nodes the new layer should contain.
        activation (str): The activation function that should be used on the
            layer.
        lambtha (float): The L2 regularization parameter.

    Returns:
        The output of the new layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode='fan_avg')
    regularizer = tf.keras.regularizers.L2(lambtha)
    new_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )
    return new_layer(prev)
