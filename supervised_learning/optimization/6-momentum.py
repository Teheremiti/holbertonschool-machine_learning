#!/usr/bin/env python3
"""
   Momentum upgraded
"""

import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
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
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha,
                                           momentum=beta1)
    train_op = optimizer.minimize(loss)
    return train_op
