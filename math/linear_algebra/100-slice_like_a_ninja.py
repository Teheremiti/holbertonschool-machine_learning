#!/usr/bin/env python3
""" np_slice function """


def np_slice(matrix, axes={}):
    """
    Slices each specified dimension of an ndarray along the axes

    Args:
        matrix (np.ndarray): The matrix to slice
        axes (dict, optional): The axes containing th e dimensions to slice
        paired with the indexes to slice. Defaults to {}.

    Returns:
        np.ndarray: The sliced matrix
    """
    result = matrix.copy()
    for axis, indices in axes.items():
        slices = [slice(None)] * len(result.shape)
        slices[axis] = slice(*indices)
        result = result[tuple(slices)]
    return result
