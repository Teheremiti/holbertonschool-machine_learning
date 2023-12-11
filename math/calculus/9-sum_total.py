#!/usr/bin/env python3
""" summation_i_squared function """


def summation_i_squared(n):
    """
    Returns the sum from 1 to n of squared integers

    Args:
        n (int): The stopping condition

    Returns:
        int: The total sum
    """
    if type(n) is not int or n < 1:
        return None

    sum = (n * (n + 1) * (2 * n + 1)) // 6
    return sum
