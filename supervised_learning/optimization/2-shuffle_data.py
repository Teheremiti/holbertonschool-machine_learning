#!/usr/bin/env python3
""" shuffle_data function """
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Args:
        X (ndarray): Matrix of shape(m, nx) to shuffle.
        Y (ndarray): Matrix of shape(m, ny) to shuffle.

    Returns:
        The shuffled X and Y matrices.
    """
    range = np.random.permutation(X.shape[0])
    return X[range], Y[range]
