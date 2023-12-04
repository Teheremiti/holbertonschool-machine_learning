#!/usr/bin/env python3
""" matrix_shape function """


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
