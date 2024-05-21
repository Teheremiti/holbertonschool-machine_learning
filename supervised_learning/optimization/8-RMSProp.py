#!/usr/bin/env python3
""" RMSProp upgraded """
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using
    the RMSProp optimization algorithm.

    Args:
        loss (float): The loss of the network.
        alpha (float): The learning rate.
        beta2 (float): The RMSProp weight.
        epsilon (float): Small number to avoid divsion by zero.

    Returns:
        The RMSProp optimization operation.
    """
    return tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                       rho=beta2,
                                       epsilon=epsilon)
