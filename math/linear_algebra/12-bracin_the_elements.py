#!/usr/bin/env python3
""" np_elementwise function """


def np_elementwise(mat1, mat2):
    """
    Computes all possible matrix operations on mat1 and mat2

    Args:
        mat1 (numpy.ndarray): The matrix to compute from
        mat2 (numpy.ndarray): The matrix to compute by

    Returns:
        tuple: The results of all the computations of mat1 and mat2 as a tuple.
               Format: (addition, substraction, multiplication, quotient).
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
