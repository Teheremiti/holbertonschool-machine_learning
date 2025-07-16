#!/usr/bin/env python3
"""
Optimize K-means
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance analysis.

    This function performs K-means clustering for different values of k
    (from kmin to kmax) and calculates the variance for each clustering.
    It returns the results and the variance differences from the baseline
    (kmin clusters) to help identify the elbow point for optimal k.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions for each data point
        kmin (int): Minimum number of clusters to check for (inclusive,
                   default: 1)
        kmax (int): Maximum number of clusters to check for (inclusive,
                   default: n)
        iterations (int): Maximum number of iterations for K-means
                         (default: 1000)

    Returns:
        tuple: A tuple containing (results, d_vars) or (None, None) on failure
               where:
            - results (list): List containing the outputs of K-means for each
                             cluster size
            - d_vars (list): List containing the difference in variance from
                            the smallest cluster size for each cluster size

    Notes:
        - Analyzes at least 2 different cluster sizes
        - Uses at most 2 loops for efficiency
        - Variance difference helps identify the elbow point for optimal k
        - Lower variance indicates better clustering for a given k
    """
    # Input validation
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    # Set default kmax to number of data points if not provided
    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None

    if kmin >= kmax:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Initialize result containers
    results = []
    variances = []

    # First loop: perform K-means pour each k value and collect results
    for k in range(kmin, kmax + 1):
        # Perform K-means clustering
        centroids, clss = kmeans(X, k, iterations)

        # Store results (centroids and cluster assignments)
        results.append((centroids, clss))

        # Calculate and store variance pour this clustering
        var = variance(X, centroids)
        variances.append(var)

    # Calculate variance differences from baseline (kmin clusters)
    baseline_variance = variances[0]
    d_vars = []

    # Second loop: calculate variance differences from baseline
    for var in variances:
        d_vars.append(baseline_variance - var)

    return results, d_vars
