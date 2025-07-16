#!/usr/bin/env python3
"""
Maximization step for EM algorithm in GMM.
"""

import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.

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
        - Posterior probabilities should sum to 1 for each data point
        - Covariance matrices are computed using weighted outer products
    """
    # Input validation - check array types and dimensions
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    # Get dimensions
    n, d = X.shape
    k, g_n = g.shape

    # Check dimension compatibility
    if g_n != n:
        return None, None, None

    # Validate that posterior probabilities are properly normalized
    # Each data point's probabilities should sum to approximately 1
    col_sums = np.sum(g, axis=0)
    if not np.allclose(col_sums, 1.0, rtol=1e-10):
        return None, None, None

    # Calculate effective sample sizes for each cluster
    # N_k = Σ_i γ(z_k_i)
    effective_sample_sizes = np.sum(g, axis=1)

    # Check for empty clusters (avoid division by zero)
    if np.any(effective_sample_sizes <= 0):
        return None, None, None

    # Initialize parameter arrays
    pi = np.zeros(k)
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    # Update parameters for each cluster (using 1 loop)
    for cluster_idx in range(k):
        # Extract responsibilities for this cluster
        responsibilities = g[cluster_idx]  # shape: (n,)
        N_k = effective_sample_sizes[cluster_idx]

        # Update prior: π_k = N_k / n
        pi[cluster_idx] = N_k / n

        # Update mean: μ_k = (Σ_i γ(z_k_i) * x_i) / N_k
        # Using matrix multiplication for efficiency
        m[cluster_idx] = np.dot(responsibilities, X) / N_k

        # Update covariance matrix:
        # Σ_k = (Σ_i γ(z_k_i) * (x_i - μ_k)(x_i - μ_k)ᵀ) / N_k
        # Center the data points
        centered_X = X - m[cluster_idx]  # shape: (n, d)

        # Compute weighted covariance using outer products
        # Σ_k = (1/N_k) * Σ_i γ(z_k_i) * (x_i - μ_k) * (x_i - μ_k)ᵀ
        # Using einsum for efficient computation: 'i,ij,ik->jk'
        # i: data points, j,k: dimensions
        S[cluster_idx] = np.einsum(
            'i,ij,ik->jk', responsibilities, centered_X, centered_X) / N_k

    return pi, m, S
