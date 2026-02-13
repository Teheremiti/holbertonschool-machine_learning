#!/usr/bin/env python3
"""Policy Gradient"""

import numpy as np


def policy(matrix, weight):
    """
    Compute the policy (action probabilities) given state matrix and weights.

    Args:
        matrix: State matrix.
        weight: Weight array to apply in the policy.

    Returns:
        Matrix of probabilities for each possible action.
    """
    z = matrix @ weight
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Compute the Monte-Carlo policy gradient from state and weight matrix.

    Args:
        state: Matrix representing the current observation of the environment.
        weight: Matrix of random weights.

    Returns:
        The action (int) and the gradient (ndarray), in this order.
    """
    state = np.atleast_2d(state)
    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs[0])
    one_hot = np.zeros_like(probs)
    one_hot[0, action] = 1
    grad = state.T @ (one_hot - probs)
    return action, grad
