#!/usr/bin/env python3
""" np_matmul function """

import numpy as np


def np_matmul(mat1, mat2):
    """
    Multiplies two numpy.ndarrays

    Args:
        mat1 (np.ndarray): The matrix to mulitply
        mat2 (np.ndarray): The matrix to mulitply by

    Returns:
        np.ndarray: The product of mat1 and mat2
    """
    return np.dot(mat1, mat2)
