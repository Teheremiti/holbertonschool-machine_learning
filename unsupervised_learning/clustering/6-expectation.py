#!/usr/bin/env python3
"""
Expectation in EM algorithm for GMM.
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.

    The expectation step computes the posterior probabilities
    (responsibilities) of each data point belonging to each cluster using
    Bayes' theorem:
    γ(z_k) = π_k * N(x | μ_k, Σ_k) / Σ_j π_j * N(x | μ_j, Σ_j)

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions
        pi (numpy.ndarray): Prior probabilities of shape (k,) for each cluster
        m (numpy.ndarray): Cluster means of shape (k, d) for each cluster
        S (numpy.ndarray): Covariance matrices of shape (k, d, d) for each
                          cluster

    Returns:
        tuple: A tuple containing (g, l) or (None, None) on failure where:
            - g (numpy.ndarray): Posterior probabilities of shape (k, n) where
                                g[i, j] = P(cluster i | data point j)
            - l (float): Total log likelihood of the data given the model

    Notes:
        - Uses at most 1 loop for efficiency
        - Posterior probabilities sum to 1 for each data point
        - Log likelihood measures how well the model fits the data
    """
    # Input validation - check array types and dimensions
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None

    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None

    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    # Get dimensions
    n, d = X.shape
    k = pi.shape[0]

    # Check dimension compatibility
    if m.shape != (k, d):
        return None, None

    if S.shape != (k, d, d):
        return None, None

    # Check if priors sum to 1 (valid probability distribution)
    if not np.isclose(np.sum(pi), 1.0):
        return None, None

    # Initialize posterior probabilities matrix
    g = np.zeros((k, n))

    # Calculate weighted likelihoods pour each cluster (using 1 loop)
    for cluster_idx in range(k):
        # Calculate likelihood P(X | cluster_idx) pour all data points
        likelihood = pdf(X, m[cluster_idx], S[cluster_idx])

        # Check if pdf calculation failed
        if likelihood is None:
            return None, None

        # Calculate weighted likelihood: π_k * P(X | μ_k, Σ_k)
        g[cluster_idx] = pi[cluster_idx] * likelihood

    # Calculate marginal likelihood P(X) pour normalization
    # This is the sum over all clusters pour each data point
    marginal_likelihood = np.sum(g, axis=0)

    # Check pour numerical issues (zero marginal likelihood)
    if np.any(marginal_likelihood <= 0):
        return None, None

    # Normalize to get posterior probabilities using Bayes' theorem
    # γ(z_k) = π_k * P(X | μ_k, Σ_k) / P(X)
    g = g / marginal_likelihood

    # Calculate total log likelihood
    # L = Σ_n log(P(x_n)) = Σ_n log(Σ_k π_k * P(x_n | μ_k, Σ_k))
    log_likelihood = np.sum(np.log(marginal_likelihood))

    return g, log_likelihood
