#!/usr/bin/env python3
"""Absorbing Chains"""

import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing.

    Args:
        P (numpy.ndarray): shape (n, n) standard transition matrix

    Returns:
        True if absorbing, False otherwise or on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    n, m = P.shape
    if n != m:
        return False

    # Step 1: Find absorbing states (P[i, i] == 1 and row all zeros except i)
    absorbing_states = np.isclose(
        np.diag(P),
        1) & np.isclose(
        P.sum(
            axis=1) -
        np.diag(P),
        0)
    if not np.any(absorbing_states):
        return False

    # Step 2: Check if every state can reach an absorbing state
    reachable = P.copy()
    for _ in range(n):
        reachable = np.matmul(reachable, P)

    # If for each state i, there exists a path to an absorbing state j
    for i in range(n):
        if not np.any(reachable[i][absorbing_states]):
            return False

    return True
