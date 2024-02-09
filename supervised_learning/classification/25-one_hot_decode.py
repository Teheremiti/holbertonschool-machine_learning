#!/usr/bin/env python3
""" one_hot_encode function"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot vector into a numerical label vector.

    one_hot (ndarray): one-hot encoded matrix with shape (classes, m), with m
        the maximum number of classes.

    Returns:
        An ndarray with shape (m, ) containing the numeric labels for each
        example, or None on failure.
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)
