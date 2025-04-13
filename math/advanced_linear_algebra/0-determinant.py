#!/usr/bin/env python3
"""Determinant"""


def determinant(matrix):
    """
    Calculates the determinant of a square matrix using Laplace expansion.

    Args:
    matrix (list[]): The matrix whose determinant should be calculated.

    Returns:
    The determinant of the matrix.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix[0]) == 0:
        return 1

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    determinant_result = 0
    for i in range(len(matrix[0])):
        minor = [row[:i] + row[i + 1:] for row in matrix[1:]]
        cofactor = (-1) ** i * matrix[0][i]
        determinant_result += cofactor * determinant(minor)

    return determinant_result
