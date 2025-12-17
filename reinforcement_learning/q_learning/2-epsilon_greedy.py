#!/usr/bin/env python3
"""Epsilon-greedy action selection for Q-learning."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Select an action using the epsilon-greedy strategy.

    With probability ``epsilon``, the policy explores by sampling a random
    action uniformly from the available actions. Otherwise, it exploits the
    current knowledge by choosing the action with the highest Q-value for the
    given state.

    Args:
        Q: NumPy array representing the Q-table. ``Q[s, a]`` is the estimated
            value of taking action ``a`` in state ``s``.
        state: Current state index used to index into ``Q``.
        epsilon: Exploration probability in :math:`[0, 1]`. Larger values lead
            to more random exploration.

    Returns:
        The index of the selected action.
    """
    probability = np.random.rand()

    if probability < epsilon:
        action = np.random.randint(0, Q.shape[1])
    else:
        action = np.argmax(Q[state, :])

    return action
