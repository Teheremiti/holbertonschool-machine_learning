#!/usr/bin/env python3
"""The Forward Algorithm"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden Markov model.

    Args:
        Observation (numpy.ndarray): shape (T,), index of observations
        Emission (numpy.ndarray): shape (N, M), emission probabilities
        Transition (numpy.ndarray): shape (N, N), transition probabilities
        Initial (numpy.ndarray): shape (N, 1), starting probabilities

    Returns:
        P (float): likelihood of the observations
        F (numpy.ndarray): shape (N, T), forward path probabilities
    """
    if (not isinstance(Observation, np.ndarray) or
        len(Observation.shape) != 1 or
        not isinstance(Emission, np.ndarray) or
        not isinstance(Transition, np.ndarray) or
            not isinstance(Initial, np.ndarray)):
        return None, None

    T = Observation.shape[0]
    N, M = Emission.shape

    if (Transition.shape != (N, N) or
            Initial.shape != (N, 1)):
        return None, None

    F = np.zeros((N, T))

    # Initialization step
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Recursion step
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.dot(F[:, t - 1], Transition[:, j]) * \
                Emission[j, Observation[t]]

    # Termination step
    P = np.sum(F[:, -1])

    return P, F
