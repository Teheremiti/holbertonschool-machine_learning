#!/usr/bin/env python3
"""Adjugate"""

cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a square matrix.

    Args:
    matrix (list[]): Matrix whose adjugate matrix should be calculated.

    Returns:
    The adjugate matrix of the input matrix.
    """
    n = len(matrix)
    if not isinstance(matrix, list) or n == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) == 0 or len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    cof = cofactor(matrix)
    adjugate_matrix = [list(row) for row in zip(*cof)]
    return adjugate_matrix
