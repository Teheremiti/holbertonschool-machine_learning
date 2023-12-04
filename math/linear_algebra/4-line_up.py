#!/usr/bin/env python3
""" add_arrays function """


def add_arrays(arr1, arr2):
    """
    Adds up the elements of two arrays

    Args:
        arr1 (array of ints/floats): Array to add
        arr2 (array of ints/floats): Array to add

    Returns:
        list: The new list of summed elements
    """
    if len(arr1) != len(arr2):
        return None

    sum = []
    for i in range(len(arr1)):
        sum.append(arr1[i] + arr2[i])
    return sum
