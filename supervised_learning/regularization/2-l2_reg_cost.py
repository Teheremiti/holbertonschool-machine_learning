#!/usr/bin/env python3
""" L2 regularization cost """
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost (tensor): The cost of the network without L2 regularization.

    Returns:
        tensor: The cost of the network accounting for L2 regularization.
    """
    return cost + tf.compat.v1.losses.get_regularization_losses()
