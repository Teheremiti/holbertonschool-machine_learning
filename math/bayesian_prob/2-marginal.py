#!/usr/bin/env python3
"""Marginal"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of observing x patients with severe side effects
    out of n total patients for each probability in P, assuming a binomial
    distribution.

    Args:
    x (int): Number of patients that develop severe side effects.
    n (int): Total number of patients observed.
    P (np.ndarray): 1D array of hypothetical probabilities of developing severe
    side effects.

    Returns:
    np.ndarray: The 1D array of likelihoods corresponding to each probability
    in P.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    binom_coeff = (np.math.factorial(n)
                   / (np.math.factorial(x) * np.math.factorial(n - x)))

    # Likelihood: binom_coeff * P^x * (1-P)^(n-x)
    likelihoods = binom_coeff * (P ** x) * ((1 - P) ** (n - x))
    return likelihoods


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining this data with the various
    hypothetical probabilities.

    Args:
    x (int): Number of patients that develop severe side effects.
    n (int): Total number of patients observed.
    P (np.ndarray): 1D arraycontaining the various hypothetical probabilities
    of developing severe side effects.
    Pr (np.ndarray): 1D array containing the prior beliefs of P.

    Returns:
    np.ndarray: The 1D array containing the intersection of obtaining x and n
    with each probability P, respectively.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    likelihoods = likelihood(x, n, P)
    intersections = likelihoods * Pr
    return intersections


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability.

    Args:
    x (int): number of patients that develop severe side effects
    n (int): total number of patients observed
    P (np.ndarray): 1D array containing the various hypothetical probabilities
    of patients developing severe side effects.
    Pr (np.ndarray): 1D array containing the prior beliefs about P.

    Returns
    float: The marginal probability of obtaining x and n.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    intersections = intersection(x, n, P, Pr)
    marginals = np.sum(intersections)
    return marginals
