#!/usr/bin/env python3
"""
Principal Component Analysis (PCA) module.

This module provides functionality to perform PCA on a dataset while
maintaining a specified fraction of the original variance.
"""
import numpy as np


def pca(X, var=0.95):
    """
    Perform PCA on a dataset to maintain a specified fraction of variance.

    Principal Component Analysis (PCA) is a dimensionality reduction technique
    that projects data onto a lower-dimensional subspace while preserving as
    much variance as possible. This function uses Singular Value Decomposition
    (SVD) to find the principal components efficiently.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, d)
        Input dataset where:
        - n is the number of data points
        - d is the number of dimensions in each point
        - All dimensions have a mean of 0 across all data points
    var : float, default=0.95
        The fraction of the variance that the PCA transformation should maintain.
        Must be between 0 and 1.

    Returns
    -------
    numpy.ndarray of shape (d, nd)
        The weights matrix W that maintains the specified fraction of X's
        original variance, where nd is the new dimensionality.

    Notes
    -----
    The function assumes that the input data X is already centered (mean = 0).
    Uses SVD for numerical stability and efficiency.
    """
    # Perform Singular Value Decomposition
    # U: left singular vectors, S: singular values, V: right singular vectors
    _, S, V = np.linalg.svd(X, full_matrices=False)

    # Calculate explained variance ratio for each component
    # Variance is proportional to the square of singular values
    explained_variance_ratio = (S ** 2) / np.sum(S ** 2)

    # Calculate cumulative explained variance ratio
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Find the minimum number of components needed to maintain desired variance
    num_components = np.argmax(cumulative_variance >= var) + 1

    # Extract the transformation matrix (first num_components principal axes)
    # V is already transposed from SVD, so we take the first num_components rows
    # and transpose to get the weights matrix of shape (d, nd)
    W = V[:num_components].T

    return W
