#!/usr/bin/env python3
"""
Bayesian Information Criterion (BIC) for GMM selection.
"""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion.

    The BIC balances model fit (log likelihood) with model complexity (number
    of parameters). Lower BIC values indicate better models. The formula used
    is:
    BIC = p * ln(n) - 2 * l
    where p is the number of parameters, n is the number of data points, and
    l is log likelihood.

    For a GMM with k clusters and d dimensions, the number of parameters is:
    p = k*(1 + d + d*(d+1)/2) - 1
    - (k-1) prior parameters (they sum to 1)
    - k*d mean parameters
    - k*d*(d+1)/2 covariance parameters (symmetric matrices)

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) containing the data set
        kmin (int): Positive integer containing the minimum number of clusters
                   to check for (inclusive, default: 1)
        kmax (int): Positive integer containing the maximum number of clusters
                   to check for (inclusive, default: None = max possible)
        iterations (int): Positive integer containing the maximum number of
                         iterations for the EM algorithm (default: 1000)
        tol (float): Non-negative float containing tolerance for the EM
                    algorithm (default: 1e-5)
        verbose (bool): Boolean that determines if the EM algorithm should
                       print information to standard output (default: False)

    Returns:
        tuple: A tuple containing (best_k, best_result, l, b) or (None, None,
               None, None) on failure where:
            - best_k (int): The best value for k based on its BIC
            - best_result (tuple): Tuple containing (pi, m, S) for the best k
            - l (numpy.ndarray): Array of shape (kmax - kmin + 1) containing
                                log likelihood for each k
            - b (numpy.ndarray): Array of shape (kmax - kmin + 1) containing
                                BIC value for each k

    Notes:
        - Uses at most 1 loop to test different k values
        - Lower BIC values indicate better model selection
        - Automatically handles model complexity vs fit trade-off
    """
    # Input validation
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    # Set kmax to maximum possible clusters if not specified
    n, d = X.shape
    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None

    if kmin > kmax:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    # Initialize result arrays
    num_tests = kmax - kmin + 1
    log_likelihoods = np.zeros(num_tests)
    bic_values = np.zeros(num_tests)
    em_results = []

    # Test each value of k (using 1 loop)
    for i, k in enumerate(range(kmin, kmax + 1)):
        # Run EM algorithm for current k
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        # Check if EM algorithm failed
        if pi is None:
            return None, None, None, None

        # Store EM results
        em_results.append((pi, m, S))
        log_likelihoods[i] = log_likelihood

        # Calculate number of parameters for GMM with k clusters and d
        # dimensions
        # p = (k-1) + k*d + k*d*(d+1)/2
        # = k*(1 + d + d*(d+1)/2) - 1
        num_priors = k - 1  # Priors sum to 1, so only k-1 free parameters
        num_means = k * d   # k mean vectors, each with d components
        num_covariances = k * d * (d + 1) // 2  # k symmetric d×d matrices
        total_parameters = num_priors + num_means + num_covariances

        # Calculate BIC: BIC = p * ln(n) - 2 * l
        bic_values[i] = total_parameters * np.log(n) - 2 * log_likelihood

    # Find the best k (minimum BIC)
    best_idx = np.argmin(bic_values)
    best_k = kmin + best_idx
    best_result = em_results[best_idx]

    return best_k, best_result, log_likelihoods, bic_values
