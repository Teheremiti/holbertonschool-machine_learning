#!/usr/bin/env python3
"""Mean and Covariance"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Args:
    X (np.ndarray): Data set of shape (n, d), with `n` and `d` the numbers of
    data points and dimensions in each data point, respectively.

    Returns:
    The mean and and covariance of the input data set:
        - mean: `np.ndarray` of shape (1, d)
        - cov: `np.ndarray` of shape (d, d)
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    centered = X - mean
    cov = np.dot(centered.T, centered) / (n - 1)

    return mean, cov
