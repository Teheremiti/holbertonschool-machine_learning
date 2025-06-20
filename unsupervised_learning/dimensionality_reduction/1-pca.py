#!/usr/bin/env python3
"""
PCA improved
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset and returns the data transformed to a new
    dimensionality.

    Args:
        X (numpy.ndarray): Matrix of shape (n, d) where n is the number of
            data points and d is the number of original dimensions. It is
            assumed that all dimensions have comparable scale, but they do
            not need to be centered; this function centers them internally.
        ndim (int): Desired number of dimensions for the transformed data.

    Returns:
        numpy.ndarray: Transformed data `T` of shape (n, ndim).

    Notes:
        • This implementation uses Singular Value Decomposition (SVD) which is
          numerically stable and avoids the explicit construction of the
          covariance matrix.
        • Only the first ``ndim`` right-singular vectors are used, providing
          the directions of maximum variance.
    """
    # Validate ndim
    if not isinstance(ndim, int) or ndim <= 0:
        raise ValueError("ndim must be a positive integer")

    # Center the dataset (zero mean per feature)
    X_centered = X - np.mean(X, axis=0)

    # Compute the compact SVD of the centered data
    # X_centered = U * S * Vt, where rows of Vt are principal directions
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)

    if ndim > Vt.shape[0]:
        raise ValueError("ndim cannot be greater than the number of features")

    # Project the data onto the first `ndim` principal components
    # Vt[:ndim] has shape (ndim, d); its transpose is (d, ndim)
    T = X_centered @ Vt[:ndim].T

    return T
