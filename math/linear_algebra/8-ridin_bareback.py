#!/usr/bin/env python3
""" mat_mul function """


def mat_mul(mat1, mat2):
    """_summary_

    Args:
        mat1 (2D-list): List to multiply
        mat2 (2D-list): List to multiply by

    Returns:
        2D-list: The multiple of mat1 and mat2
    """
    if len(mat1[0]) != len(mat2):
        return None

    mul = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            sum = 0
            for k in range(len(mat2)):
                sum += mat1[i][k] * mat2[k][j]
            row.append(sum)
        mul.append(row)
    return mul
