#!/usr/bin/env python3
"""
Variance
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a dataset.

    The intra-cluster variance is the sum of squared distances from each data
    point to its nearest cluster centroid. This metric measures how tightly
    clustered the data points are around their respective centroids.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions for each data point
        C (numpy.ndarray): Centroid means of shape (k, d) where:
            - k is the number of clusters
            - d is the number of dimensions (must match X)

    Returns:
        float: The total intra-cluster variance, or None on failure

    Notes:
        - Uses vectorized operations without loops for efficiency
        - Each point contributes the squared distance to its closest centroid
        - Lower variance indicates tighter clustering
    """
    # Input validation - check if inputs are numpy arrays
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None

    # Validate array dimensions - both must be 2D
    if X.ndim != 2 or C.ndim != 2:
        return None

    # Validate dimension compatibility - same number of features
    if X.shape[1] != C.shape[1]:
        return None

    # Calculate distances from each data point to each centroid
    # Broadcasting: X.shape = (n, d), C.shape = (k, d)
    # X[:, np.newaxis, :] creates shape (n, 1, d)
    # C[np.newaxis, :, :] creates shape (1, k, d)
    # Result: distances.shape = (n, k) - distance from each point to each
    # centroid
    distances = np.linalg.norm(
        X[:, np.newaxis, :] - C[np.newaxis, :, :], axis=2)

    # Find minimum distance pour each data point (distance to closest centroid)
    min_distances = np.min(distances, axis=1)

    # Calculate total intra-cluster variance (sum of squared minimum distances)
    total_variance = np.sum(min_distances ** 2)

    return total_variance
