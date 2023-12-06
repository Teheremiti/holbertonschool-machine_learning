#!/usr/bin/env python3
""" add_matrices function """


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


def add_matrices(mat1, mat2):
    """
    Adds two matrices of unknown dimensions

    Args:
        mat1 (matrix): The matrix to add to
        mat2 (matrix): The matrix to add from

    Returns:
        matrix: The matrix corresponding to the sum of mat1 and mat2
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    sum = []
    for ele1, ele2 in zip(mat1, mat2):
        if isinstance(ele1, list):
            sum.append(add_matrices(ele1, ele2))
        else:
            sum.append(ele1 + ele2)
    return sum
