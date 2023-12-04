#!/usr/bin/env python3
""" cat_matrices2D function """


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices

    Args:
        mat1 (2D list): 2D list to concatenate into
        mat2 (2D list): 2D list to concatenate from
        axis (int, optional): Axis to contenate along. Defaults to 0.

    Returns:
        2D list: The concatenated matrix
    """
    concat = []
    if axis == 0:
        concat = mat1 + mat2
    else:
        for i in range(len(mat2)):
            concat.append(mat1[i] + mat2[i])
    return concat
