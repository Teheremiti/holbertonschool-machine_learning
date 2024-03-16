#!/usr/bin/env python3
""" L2 regularization cost """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.
    FORMULA: cost = loss + ((lambtha / 2m) * sum(w**2))

    Args:
        cost (float): The cost of the network without L2 regularization.
        lambtha (float): The regularization parameter.
        weights (dict): Dictionary of the weights and biases (numpy.ndarrays)
            of the neural network.
        L (int): The number of layers in the neural network.
        m (int): The number of data points used.

    Returns:
        ndarray: The cost of the network accounting for L2 regularization.
    """
    sum = 0
    for i in range(1, L + 1):
        sum += np.sum(np.square(weights[f'W{i}']))

    L2_cost = cost + (lambtha / (2 * m)) * sum
    return L2_cost
