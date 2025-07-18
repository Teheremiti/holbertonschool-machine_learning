#!/usr/bin/env python3
"""
Complete EM algorithm for GMM.
"""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization algorithm for a Gaussian Mixture
    Model.

    The EM algorithm iteratively performs two steps:
    1. E-step: Compute posterior probabilities using current parameters
    2. M-step: Update parameters using computed posterior probabilities

    The algorithm continues until convergence (log likelihood change ≤
    tolerance) or maximum iterations are reached.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions
        k (int): Positive integer containing the number of clusters
        iterations (int): Positive integer containing the maximum number of
                         iterations for the algorithm (default: 1000)
        tol (float): Non-negative float containing tolerance of the log
                    likelihood, used for early stopping (default: 1e-5)
        verbose (bool): Boolean that determines if information about the
                       algorithm should be printed (default: False)

    Returns:
        tuple: A tuple containing (pi, m, S, g, l) or (None, None, None, None,
               None) on failure where:
            - pi (numpy.ndarray): Array of shape (k,) containing the priors
                                 for each cluster
            - m (numpy.ndarray): Array of shape (k, d) containing the centroid
                                means for each cluster
            - S (numpy.ndarray): Array of shape (k, d, d) containing the
                                covariance matrices for each cluster
            - g (numpy.ndarray): Array of shape (k, n) containing the
                                probabilities for each data point in each
                                cluster
            - l (float): The log likelihood of the model

    Notes:
        - Uses at most 1 loop for the EM iterations
        - Prints log likelihood every 10 iterations if verbose=True
        - Stops early if log likelihood improvement ≤ tolerance
        - Returns final parameters after convergence or max iterations
    """
    # Input validation
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None

    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    # Get dataset dimensions
    n, d = X.shape

    # Initialize GMM parameters
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    # Initial expectation step
    g, likelihood = expectation(X, pi, m, S)
    if g is None or likelihood is None:
        return None, None, None, None, None

    likelihood_prev = 0

    # EM algorithm main loop
    for iteration in range(iterations):
        # Print log likelihood every 10 iterations if verbose
        if verbose and iteration % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                iteration, round(likelihood, 5)))

        # Maximization step: Update parameters using current posterior
        # probabilities
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        # Expectation step: Compute new posterior probabilities using
        # updated parameters
        g, likelihood = expectation(X, pi, m, S)
        if g is None or likelihood is None:
            return None, None, None, None, None

        # Check convergence (early stopping)
        diff = abs(likelihood - likelihood_prev)
        if diff <= tol:
            break

        # Update previous log likelihood pour next iteration
        likelihood_prev = likelihood

    # Print final log likelihood if verbose
    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            iteration + 1, round(likelihood, 5)))

    return pi, m, S, g, likelihood
