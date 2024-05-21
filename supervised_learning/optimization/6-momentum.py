#!/usr/bin/env python3
""" Momentum upgraded """
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Creates the training operation for a neural network in tensorflow using
    gradient descent with momentum optimization algorithm.

    Args:
        loss (float): The loss of network.
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.

    Returns:
        The momentum optimization operation.
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha,
                                   momentum=beta1)
