#!/usr/bin/env python3
"""Correlation"""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix from a covariance matrix.

    Args:
    C (np.ndarray): Covariance matrix of shape (d, d), where d is the number
    of dimensions.

    Returns:
    The correlation matrix, `np.ndarray` of shape (d, d).
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std_dev = np.sqrt(np.diag(C))
    if np.any(std_dev == 0):
        raise ValueError("Standard deviation cannot be zero on the diagonal")

    correlation = C / np.outer(std_dev, std_dev)
    return correlation
