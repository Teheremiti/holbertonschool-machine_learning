#!/usr/bin/env python3
"""
    PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset and returns the weights matrix that maintains
    a specified fraction of the original variance.

    Args:
        X (numpy.ndarray): Array of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions in each point
            - all dimensions have a mean of 0 across all data points
        var (float): The fraction of the variance that the PCA transformation
            should maintain (default: 0.95)

    Returns:
        numpy.ndarray: The weights matrix W of shape (d, nd) where nd is the
            new dimensionality that maintains var fraction of X's original
            variance
    """
    # Perform SVD decomposition of the input data
    # full_matrices=False ensures efficient computation for tall matrices
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute squared singular values (proportional to eigenvalues)
    # This represents the variance explained by each component
    S_squared = S * S

    # Calculate total variance and cumulative variance ratios efficiently
    total_variance = np.sum(S_squared)
    cumulative_var_ratio = np.cumsum(S_squared) / total_variance

    # Find the minimum number of components needed to maintain desired variance
    # argmax returns the first index where the condition is True
    n_components = np.argmax(cumulative_var_ratio >= var) + 1

    # Extract principal components, taking one extra to ensure we exceed
    # threshold. This guarantees we maintain at least the requested fraction
    # of variance
    W = Vt[:n_components + 1].T

    return W
