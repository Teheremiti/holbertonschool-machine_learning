#!/usr/bin/env python3
"""Regular Chains"""

import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular Markov chain.

    Args:
        P (numpy.ndarray): square matrix (n x n) representing transition
        probabilities

    Returns:
        numpy.ndarray of shape (1, n) with the steady state, or None on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n, m = P.shape
    if n != m:
        return None

    # Check if it's a regular Markov chain: P^k has all positive entries for
    # some k
    try:
        Pk = np.copy(P)
        for _ in range(1, 101):
            Pk = Pk @ P
            if np.all(Pk > 0):
                break
        else:
            return None  # Not regular
    except Exception:
        return None

    # Solve steady state: πP = π and sum(π) = 1  → π(P - I) = 0
    try:
        A = P.T - np.eye(n)
        A = np.vstack([A, np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1
        steady = np.linalg.lstsq(A, b, rcond=None)[0]
        return np.array([steady])
    except Exception:
        return None
