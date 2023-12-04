#!/usr/bin/env python3
""" matrix_transpose function """


def matrix_transpose(matrix):
    """
    Transposes a matrix

    Args:
        matrix (2Dimensional list): The matrix to transpose

    Returns:
        list: A new matrix corresponding to the transpose of `matrix`
    """
    m = matrix
    transposed = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
    return transposed
