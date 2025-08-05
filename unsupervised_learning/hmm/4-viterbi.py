#!/usr/bin/env python3
"""The Viterbi Algorithm"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden Markov
    model.

    Args:
        Observation (numpy.ndarray): shape (T,), index of observations
        Emission (numpy.ndarray): shape (N, M), emission probabilities
        Transition (numpy.ndarray): shape (N, N), transition probabilities
        Initial (numpy.ndarray): shape (N, 1), initial state probabilities

    Returns:
        path (list): length T, most likely sequence of hidden states
        P (float): probability of the path
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

    V = np.zeros((N, T))       # Viterbi matrix
    backpointer = np.zeros((N, T), dtype=int)

    # Initialization step
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Dynamic programming forward
    for t in range(1, T):
        for j in range(N):
            probs = V[:, t - 1] * Transition[:, j]
            backpointer[j, t] = np.argmax(probs)
            V[j, t] = np.max(probs) * Emission[j, Observation[t]]

    # Backtrace
    path = [np.argmax(V[:, T - 1])]
    for t in range(T - 1, 0, -1):
        path.insert(0, backpointer[path[0], t])

    P = np.max(V[:, T - 1])
    return path, P
