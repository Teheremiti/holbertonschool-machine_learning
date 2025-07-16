#!/usr/bin/env python3
"""
Maximization step in EM algorithm in GMM.
"""

import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm in a GMM.

    The maximization step updates the model parameters (priors, means,
    covariances) using the posterior probabilities computed in the expectation
    step:
    - π_k = (1/n) * Σ_i γ(z_k_i)
    - μ_k = (Σ_i γ(z_k_i) * x_i) / (Σ_i γ(z_k_i))
    - Σ_k = (Σ_i γ(z_k_i) * (x_i - μ_k)(x_i - μ_k)ᵀ) / (Σ_i γ(z_k_i))

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions
        g (numpy.ndarray): Posterior probabilities of shape (k, n) where:
            - k is the number of clusters
            - g[i, j] = P(cluster i | data point j)

    Returns:
        tuple: A tuple containing (pi, m, S) or (None, None, None) on failure
               where:
            - pi (numpy.ndarray): Updated priors of shape (k,) for each cluster
            - m (numpy.ndarray): Updated means of shape (k, d) for each cluster
            - S (numpy.ndarray): Updated covariance matrices of shape (k, d, d)
                                for each cluster

    Notes:
        - Uses at most 1 loop for efficiency
        - Posterior probabilities should sum to n across all data points
        - Covariance matrices are computed using weighted outer products
    """
    # Input validation - check array types and dimensions
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    # Get dimensions
    n, d = X.shape
    k = g.shape[0]

    # Check dimension compatibility
    if n != g.shape[1]:
        return None, None, None

    # Validate that posterior probabilities sum correctly
    # Sum posterior probabilities in each point
    sum_gi = np.sum(g, axis=0)
    val_n = np.sum(sum_gi)
    # Test if sum posterior probabilities != total number of data
    if val_n != n:
        return None, None, None

    # Initialize parameter arrays
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    # Update parameters each cluster (using 1 loop)
    i = 0
    while i < k:
        # New prior
        pi[i] = 1 / n * np.sum(g[i])
        # New centroid mean
        m[i] = np.matmul(g[i], X) / np.sum(g[i])
        X_mean = X - m[i]
        # New covariance
        S[i] = np.matmul(np.multiply(g[i], X_mean.T), X_mean) / np.sum(g[i])
        i += 1

    return pi, m, S
