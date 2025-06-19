#!/usr/bin/env python3
"""
Principal Component Analysis (PCA) with fixed dimensionality module.

This module provides functionality to perform PCA on a dataset and transform
it to a specified number of dimensions.
"""
import numpy as np


def pca(X, ndim):
    """
    Perform PCA on a dataset and transform it to specified dimensions.

    Principal Component Analysis (PCA) reduces the dimensionality of data
    by projecting it onto the principal components that capture the most
    variance. This function centers the data, performs PCA, and returns
    the transformed data with the specified number of dimensions.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, d)
        Input dataset where:
        - n is the number of data points
        - d is the number of dimensions in each point
    ndim : int
        The new dimensionality of the transformed X.
        Must be less than or equal to min(n, d).

    Returns
    -------
    numpy.ndarray of shape (n, ndim)
        The transformed version of X containing the projected data
        onto the first ndim principal components.

    Notes
    -----
    The function automatically centers the data (subtracts the mean).
    Uses SVD for numerical stability and efficiency.
    """
    # Center the data by subtracting the mean along each feature
    X_centered = X - np.mean(X, axis=0)

    # Perform Singular Value Decomposition
    # U: left singular vectors, S: singular values, V: right singular vectors
    _, _, V = np.linalg.svd(X_centered, full_matrices=False)

    # Extract the transformation matrix for the first ndim components
    # V contains the principal components as rows, so we take first ndim rows
    # and transpose to get the weights matrix of shape (d, ndim)
    W = V[:ndim].T

    # Transform the centered data using the principal components
    # Project X_centered onto the principal component space
    T = X_centered @ W

    return T
