#!/usr/bin/env python3
"""
Initialize Gaussian Mixture Model (GMM)
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.

    This function sets up the initial parameters for a GMM by using K-means
    clustering to find initial centroids, setting equal prior probabilities
    for all clusters, and initializing covariance matrices as identity
    matrices.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions for each data point
        k (int): Positive integer containing the number of clusters

    Returns:
        tuple: A tuple containing (pi, m, S) or (None, None, None) on failure
               where:
            - pi (numpy.ndarray): Array of shape (k,) containing the priors
                                 for each cluster, initialized evenly
                                 (1/k each)
            - m (numpy.ndarray): Array of shape (k, d) containing the centroid
                                means for each cluster, initialized with
                                K-means
            - S (numpy.ndarray): Array of shape (k, d, d) containing the
                                covariance matrices for each cluster,
                                initialized as identity matrices

    Notes:
        - Uses K-means clustering to initialize cluster centers
        - All clusters start with equal probability (uniform priors)
        - Covariance matrices start as identity (spherical, unit variance)
        - No loops used in implementation for efficiency
    """
    # Input validation - check if X is a proper 2D numpy array
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    # Input validation - check if k is a positive integer
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    # Get dataset dimensions
    n, d = X.shape

    # Initialize priors (pi) - equal probability pour each cluster
    pi = np.full(k, 1.0 / k)

    # Initialize means (m) using K-means clustering
    m, _ = kmeans(X, k)

    # Check if K-means failed
    if m is None:
        return None, None, None

    # Initialize covariance matrices (S) as identity matrices
    # Create k identity matrices of size (d, d)
    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
