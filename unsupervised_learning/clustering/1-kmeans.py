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
            - d is the number of dimensions for each data point
        k (int): Positive integer containing the number of clusters
        iterations (int): Positive integer containing the maximum number of
                         iterations to perform (default: 1000)

    Returns:
        tuple: A tuple containing (C, clss) or (None, None) on failure where:
            - C (numpy.ndarray): Array of shape (k, d) containing the centroid
                                means for each cluster
            - clss (numpy.ndarray): Array of shape (n,) containing the index
                                   of the cluster in C that each data point
                                   belongs to

    Notes:
        - Uses numpy.random.uniform exactly twice for initialization
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

    # Find min and max values for uniform distribution
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # Initialize centroids using multivariate uniform distribution (first use)
    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    # K-means algorithm
    for iteration in range(iterations):
        # Calculate distances from each point to each centroid
        distances = np.sqrt(np.sum((X[:, np.newaxis, :] -
                                   centroids[np.newaxis, :, :]) ** 2, axis=2))

        # Assign each point to closest centroid
        clss = np.argmin(distances, axis=1)

        # Store previous centroids for convergence check
        prev_centroids = centroids.copy()

        # Update centroids
        for cluster_idx in range(k):
            # Find points assigned to this cluster
            cluster_mask = (clss == cluster_idx)

            if np.any(cluster_mask):
                # Update centroid to mean of assigned points
                centroids[cluster_idx] = np.mean(X[cluster_mask], axis=0)
            else:
                # Reinitialize centroid if no points assigned (second use)
                centroids[cluster_idx] = np.random.uniform(
                    low=min_vals, high=max_vals, size=d)

        # Check for convergence
        if np.allclose(centroids, prev_centroids):
            break

    return centroids, clss
