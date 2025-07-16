#!/usr/bin/env python3
"""
Initialize K-means
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means clustering.

    This function creates k cluster centroids by sampling from a multivariate
    uniform distribution where each dimension's range is defined by the
    minimum and maximum values of the input dataset along that dimension.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions for each data point
        k (int): Positive integer representing the number of clusters

    Returns:
        numpy.ndarray: Array of shape (k, d) containing the initialized
                      centroids for each cluster, or None on failure

    Raises:
        None: Returns None instead of raising exceptions for invalid inputs
    """
    # Validate k parameter
    if not isinstance(k, int) or k <= 0:
        return None

    # Validate X parameter
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    # Get dataset dimensions
    n, d = X.shape

    # Find minimum and maximum values along each dimension
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # Initialize centroids using multivariate uniform distribution
    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    return centroids
