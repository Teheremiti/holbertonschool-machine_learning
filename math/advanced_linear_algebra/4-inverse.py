#!/usr/bin/env python3
"""Inverse"""

determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """
    Calculates the inverse matrix of a square matrix.

    Args:
    matrix (list[]): Matrix whose inverse matrix should be calculated.

    Retunrs:
    The inverse matrix of the input matrix.
    """
    n = len(matrix)
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) == 0 or len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adj = adjugate(matrix)
    inverse_matrix = [[element / det for element in row] for row in adj]
    return inverse_matrix
