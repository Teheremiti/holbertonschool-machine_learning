#!/usr/bin/env python3
""" add_matrices function """


def add_matrices2D(mat1, mat2):
    """_summary_

    Args:
        mat1 (matrix): Matrix to add
        mat2 (matrix): Matrix to add

    Returns:
        matrix: The matrix corresponding to the addition of mat1 and mat2
    """
    l1 = len(mat1)
    l_1 = len(mat1[0])
    if l1 != len(mat2) or l_1 != len(mat2[0]):
        return None

    sum = [[mat1[i][j] + mat2[i][j] for j in range(l_1)] for i in range(l1)]
    return sum
