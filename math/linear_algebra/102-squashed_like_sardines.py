#!/usr/bin/env python3
""" cat_matrices """


def matrix_shape(matrix):
    """
    Returns the shape of a matrix as a list

    Args:
        matrix (n-dimensional array): The matrix to find the shape of

    Returns:
        list: The dimensions of the array as a list
    """
    if not matrix or not isinstance(matrix, list):
        return []

    sub_len = matrix_shape(matrix[0])
    shape = [len(matrix)] + sub_len
    return shape


def recursive_cat(mat1, mat2, current_axis, target_axis):
    """
    Recursively concatenates two matrices of inknown dimension along a specific
    axis.

    Args:
        mat1 (matrix): The matrix to concatenate into.
        mat2 (matrix): The matrix to concatenate from.
        current_axis (int: The current axis iterating through the matrix' axes.
        target_axis (int): The axis to concatenate along.

    Returns:
        matrix: The concatenation of mat1 and mat2 along axis. Is the same
                shape as mat1 and mat2.
    """
    if current_axis == target_axis:
        return mat1 + mat2

    concat = []
    for m1, m2 in zip(mat1, mat2):
        if isinstance(m1[0], list):
            concat.append(recursive_cat(m1, m2, current_axis + 1, target_axis))
        else:
            concat.append(m1 + m2)
    return concat


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices of unknown dimension along a specific axis

    Args:
        mat1 (matrix): Matrix to concatenate into
        mat2 (matrix): Matrix to concatenate from
        axis (int, optional): Axis to concatenate along. Defaults to 0.

    Returns:
        matrix: The matrix corresponding to the concatenation of mat1 and mat2
                along axis.
    """
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    if len(shape1) != len(shape2):
        return None

    for i in range(len(shape1)):
        if i == axis:
            continue
        else:
            if shape1[i] != shape2[i]:
                return None

    return recursive_cat(mat1, mat2, 0, axis)
