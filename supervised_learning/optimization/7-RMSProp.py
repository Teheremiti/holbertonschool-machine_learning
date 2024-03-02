#!/usr/bin/env python3
""" RMSProp """
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta2 (float): RMSProp weight.
        epsilon (float): Small number to avoid division by zero.
        var (ndarray): Matrix containing the variable to be updated.
        grad (ndarray):cMatrix containing the gradient of var.
        s (ndarray): previous second moment of var.

    Returns:
        The updated variable and the new moment.
    """
    squared_gradient = beta2 * s + (1 - beta2) * grad**2
    update_var = var - alpha * grad / (np.sqrt(squared_gradient) + epsilon)
    return update_var, squared_gradient
