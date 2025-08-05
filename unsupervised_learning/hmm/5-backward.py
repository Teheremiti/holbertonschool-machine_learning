#!/usr/bin/env python3
"""The Backward Algorithm"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden Markov model.

    Args:
        Observation (np.ndarray): shape (T,), index of observations
        Emission (np.ndarray): shape (N, M), emission probabilities
        Transition (np.ndarray): shape (N, N), transition probabilities
        Initial (np.ndarray): shape (N, 1), initial state probabilities

    Returns:
        P (float): likelihood of the observations given the model
        B (np.ndarray): shape (N, T), backward path probabilities
    """
    if (not isinstance(Observation, np.ndarray) or
        len(Observation.shape) != 1 or
        not isinstance(Emission, np.ndarray) or
        not isinstance(Transition, np.ndarray) or
            not isinstance(Initial, np.ndarray)):
        return None, None

    T = Observation.shape[0]
    N, M = Emission.shape

    if Transition.shape != (N, N) or Initial.shape != (N, 1):
        return None, None

    B = np.zeros((N, T))
    B[:, -1] = 1  # Initialization: beta at final time step

    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(
                Transition[i, :] *
                Emission[:, Observation[t + 1]] *
                B[:, t + 1]
            )

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return P, B
