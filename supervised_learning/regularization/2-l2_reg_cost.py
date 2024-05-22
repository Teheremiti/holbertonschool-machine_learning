#!/usr/bin/env python3
""" L2 regularization cost """
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost (tensor): The cost of the network without L2 regularization.
        model (keras.Model): Includes layers with L2 regularization.

    Returns:
        tensor: The cost of the network accounting for L2 regularization.
    """
    return cost + model.losses
