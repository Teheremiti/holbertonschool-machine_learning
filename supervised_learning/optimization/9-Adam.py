#!/usr/bin/env python3
""" Adam """
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using Adam optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The weight for the first moment.
        beta2 (float): The weight for the second moment.
        epsilon (float): Small number to avoid division by zero.
        var (ndarray): Contains the variable to be updated.
        grad (ndarray): Contains the gradient of var.
        v (ndarray): The previous first moment of var.
        s (ndarray): The previous second moment of var.
        t (int): The time step used for bias correction.

    Returns:
        The updated variable, the new first moment, and the new second moment.
    """
    new_v = beta1 * v + (1 - beta1) * grad
    new_s = beta2 * s + (1 - beta2) * grad**2

    v_corrected = new_v / (1 - beta1**t)
    s_corrected = new_s / (1 - beta2**t)

    var = var - alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))

    return var, new_v, new_s
