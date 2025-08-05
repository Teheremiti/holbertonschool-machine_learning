#!/usr/bin/env python3
"""The Baum-Welch Algorithm"""

import numpy as np


def forward(Observations, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden Markov model.

    Args:
        Observations (np.ndarray): shape (T,), index of observations
        Emission (np.ndarray): shape (N, M), emission probabilities
        Transition (np.ndarray): shape (N, N), transition probabilities
        Initial (np.ndarray): shape (N, 1), initial state probabilities

    Returns:
        F (np.ndarray): shape (N, T), forward path probabilities
    """
    T = Observations.shape[0]
    N = Emission.shape[0]
    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(F[:, t - 1] * Transition[:, j]) * \
                Emission[j, Observations[t]]
    return F


def backward(Observations, Emission, Transition):
    """
    Performs the backward algorithm for a hidden Markov model.

    Args:
        Observations (np.ndarray): shape (T,), index of observations
        Emission (np.ndarray): shape (N, M), emission probabilities
        Transition (np.ndarray): shape (N, N), transition probabilities

    Returns:
        B (np.ndarray): shape (N, T), backward path probabilities
    """
    T = Observations.shape[0]
    N = Emission.shape[0]
    B = np.zeros((N, T))
    B[:, -1] = 1
    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(Transition[i, :] *
                             Emission[:, Observations[t + 1]] *
                             B[:, t + 1])
    return B


def compute_gamma_xi(Observations, Transition, Emission, F, B):
    """
    Computes gamma and xi values for Baum-Welch.

    Args:
        Observations (np.ndarray): shape (T,), index of observations
        Transition (np.ndarray): shape (N, N), transition probabilities
        Emission (np.ndarray): shape (N, M), emission probabilities
        F (np.ndarray): shape (N, T), forward path probabilities
        B (np.ndarray): shape (N, T), backward path probabilities

    Returns:
        gamma (np.ndarray): shape (N, T), gamma values
        xi (np.ndarray): shape (N, N, T - 1), xi values
    """
    N, T = F.shape
    xi = np.zeros((N, N, T - 1))
    for t in range(T - 1):
        denom = np.sum(
            F[:, t][:, None] *
            Transition *
            Emission[:, Observations[t + 1]][None, :] *
            B[:, t + 1][None, :]
        )
        if denom == 0:
            return None, None
        xi[:, :, t] = (
            F[:, t][:, None] *
            Transition *
            Emission[:, Observations[t + 1]][None, :] *
            B[:, t + 1][None, :]
        ) / denom

    gamma = np.sum(xi, axis=1)
    final_gamma = (F[:, -1] * B[:, -1]) / np.sum(F[:, -1] * B[:, -1])
    gamma = np.hstack((gamma, final_gamma[:, None]))

    return gamma, xi


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden Markov model.

    Args:
        Observations (np.ndarray): shape (T,), index of observations
        Transition (np.ndarray): shape (N, N), transition probabilities
        Emission (np.ndarray): shape (N, M), emission probabilities
        Initial (np.ndarray): shape (N, 1), initial state probabilities
        iterations (int): number of iterations to perform

    Returns:
        Transition (np.ndarray): shape (N, N), updated transition matrix
        Emission (np.ndarray): shape (N, M), updated emission matrix
    """
    if (not isinstance(Observations, np.ndarray) or
        not isinstance(Transition, np.ndarray) or
        not isinstance(Emission, np.ndarray) or
            not isinstance(Initial, np.ndarray)):
        return None, None

    T = Observations.shape[0]
    M, N = Emission.shape

    if (Transition.shape != (M, M) or
        Initial.shape != (M, 1) or
            not isinstance(iterations, int) or iterations < 1):
        return None, None

    for _ in range(iterations):
        F = forward(Observations, Emission, Transition, Initial)
        B = backward(Observations, Emission, Transition)
        gamma, xi = compute_gamma_xi(Observations, Transition, Emission, F, B)
        if gamma is None or xi is None:
            return None, None

        Transition = np.sum(xi, axis=2) / \
            np.sum(gamma[:, :-1], axis=1, keepdims=True)

        Emission.fill(0)
        for k in range(N):
            mask = (Observations == k)
            Emission[:, k] = np.sum(gamma[:, mask], axis=1)
        Emission /= np.sum(gamma, axis=1, keepdims=True)

    return Transition, Emission
