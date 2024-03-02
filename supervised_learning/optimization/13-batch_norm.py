#!/usr/bin/env python3
""" Batch Normalization """
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch
    normalization.

    Args:
        Z (ndarray): Matrix of shape (m, n) that should be normalized.
        gamma (ndarray): Matrix of shape (1, n) containing the scales used
            for batch normalization.
        beta (ndarray): Matrix of shape (1, n) containing the offsets used
            for batch normalization.
        epsilon (ndarray): Small number used to avoid division by zero.

    Returns:
        The normalized z matrix
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    std_dev = np.sqrt(variance + epsilon)

    Z_norm = (Z - mean) / std_dev

    scaled = gamma * Z_norm + beta

    return scaled
