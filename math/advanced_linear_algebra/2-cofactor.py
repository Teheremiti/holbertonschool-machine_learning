#!/usr/bin/env python3
"""Cofactor"""

minor = __import__('1-minor').minor


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a square matrix.

    Args:
    matrix (list[]): Matrix whose cofactor matrix should be calculated.

    Returns:
    The cofactor matrix of the input matrix.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) == 0 or len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = minor(matrix)
    size = len(minor_matrix)

    cofactor_matrix = []
    for i in range(size):
        cofactor_row = []
        for j in range(size):
            sign = (-1) ** (i + j)
            cofactor_row.append(sign * minor_matrix[i][j])
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix
