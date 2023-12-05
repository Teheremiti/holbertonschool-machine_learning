#!/usr/bin/env python3
""" np_cat function """

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two numpy.ndarrays

    Args:
        mat1 (np.ndarray): The matrix to concatenate into
        mat2 (np.ndarray): The matrix to concatenate from
        axis (int, optional): The axis to concatenate along. Defaults to 0.

    Returns:
        np.ndarray: The concatenation of mat1 and mat2 along axis
    """
    return np.concatenate((mat1, mat2), axis)
