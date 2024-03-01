#!/usr/bin/env python3
""" normalize function """
import numpy as np


def normalize(X, m, s):
    """
    Calculates the normalization of a matrix.

    Args:
        X (ndarray): Matrix of shape (d,nx) to normalize.
        d (int): The number of data points.
        nx (int): The number of features.
        m (ndarray): Matrix of shape (nx,), contains the mean of all
            features of X.
        s (ndarray): Matrix of shape (nx,), contains the std of all
            features of X.

    Returns:
        The noramlized X matrix
    """
    Z = (X - m) / s
    return Z
