#!/usr/bin/env python3
"""Minor"""

determinant = __import__('0-determinant').determinant


def minor(matrix):
    """
    Calculates the minor matrix of a square matrix.

    Args:
    matrix (list[]): Matrix whose minor matrix should be calculated.

    Returns:
    The minor matrix of the input matrix.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) != len(matrix[0]) or len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    size = len(matrix)
    minor_matrix = []

    if len(matrix) == 1:
        return [[1]]

    for i in range(size):
        row_minors = []
        for j in range(size):
            # Build submatrix by removing the i-th row and j-th column
            sub = [r[:j] + r[j + 1:] for k, r in enumerate(matrix) if k != i]
            row_minors.append(determinant(sub))
        minor_matrix.append(row_minors)

    return minor_matrix
