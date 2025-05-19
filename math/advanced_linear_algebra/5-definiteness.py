#!/usr/bin/env python3
"""Definiteness"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a square matrix.

    Args:
    matrix (list[]): Matrix whose definiteness should be calculated.

    Returns:
    The string indicating the definiteness of the input matrix:
        - Positive definite
        - Positive semi-definite
        - Negative semi-definite
        - Negative definite
        - Indefinite
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    shape = matrix.shape
    if matrix.ndim != 2 or shape[0] != shape[1] or matrix.size == 0:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvalsh(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
