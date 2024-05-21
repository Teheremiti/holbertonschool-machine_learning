#!/usr/bin/env python3
""" Adam upgraded """
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using
    the Adam optimization algorithm.

    Args:
        loss (): The loss of the network.
        alpha (): The learning rate.
        beta1 (): The weight used for the first moment.
        beta2 (): The weight used for the second moment.
        epsilon (): Small number to avoid division by zero.

    Returns:
        The Adam optimization operation.
    """
    return tf.keras.optimizers.Adam(learning_rate=alpha,
                                    beta_1=beta1,
                                    beta_2=beta2,
                                    epsilon=epsilon)
