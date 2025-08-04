#!/usr/bin/env python3
"""Markov Chain"""

import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a Markov chain being in a particular state
    after a specified number of iterations.

    Args:
        P (numpy.ndarray): shape (n, n) transition matrix
        s (numpy.ndarray): shape (1, n) starting state probabilities
        t (int): number of iterations

    Returns:
        numpy.ndarray: shape (1, n), probability of being in each state
        after t iterations, or None on failure
    """
    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None
    if len(s.shape) != 2 or s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None
    if not isinstance(t, int) or t < 0:
        return None

    try:
        result = s @ np.linalg.matrix_power(P, t)
        return result
    except Exception:
        return None
