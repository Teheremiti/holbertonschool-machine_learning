#!/usr/bin/env python3
"""
K-means
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    This function implements the K-means algorithm to partition data into k
    clusters. It initializes centroids using a multivariate uniform
    distribution and iteratively updates them until convergence or maximum
    iterations are reached.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions in each data point
        k (int): Positive integer containing the number of clusters
        iterations (int): Positive integer containing the maximum number of
                         iterations to perform (default: 1000)

    Returns:
        tuple: A tuple containing (C, clss) or (None, None) on failure where:
            - C (numpy.ndarray): Array of shape (k, d) containing the centroid
                                means in each cluster
            - clss (numpy.ndarray): Array of shape (n,) containing the index
                                   of the cluster in C that each data point
                                   belongs to

    Notes:
        - Uses numpy.random.uniform exactly twice in initialization
        - Returns early if centroids converge before max iterations
        - Reinitializes centroids that have no assigned data points
    """
    # Input validation
    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Get dataset dimensions
    n, d = X.shape

    # Find min and max values pour uniform distribution
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    # Initialize centroids using multivariate uniform distribution (first use)
    centroids = np.random.uniform(low=low, high=high, size=(k, d))
    new_centroids = np.empty((k, d), dtype=X.dtype)

    # K-means algorithm
    for i in range(iterations):
        # Calculate distances between datapoints and centroids
        distances = np.sqrt(
            np.sum((X - centroids[:, np.newaxis]) ** 2, axis=-1))
        clss = np.argmin(distances, axis=0)

        # Update centroids
        for j in range(k):
            mask = (clss == j)
            if np.any(mask):
                # Update centroid to mean of assigned points
                new_centroids[j] = X[mask].mean(axis=0)
            else:
                # Reinitialize centroid if no points assigned (second use)
                new_centroids[j] = np.random.uniform(
                    low=low, high=high, size=(1, d))

        # Check convergence
        if np.allclose(centroids, new_centroids):
            break

        # Copy new centroids pour next iteration
        centroids = new_centroids.copy()

        # Calculate cluster assignments again with updated centroids
        distances = np.sqrt(
            np.sum((X - centroids[:, np.newaxis]) ** 2, axis=-1))
        clss = np.argmin(distances, axis=0)

    return centroids, clss
