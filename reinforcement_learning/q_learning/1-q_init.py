#!/usr/bin/env python3
"""Initialization utilities for Q-learning."""
import numpy as np


def q_init(env):
    """Initialize a Q-table for a discrete Gym environment.

    The Q-table has one row per observation and one column per action. All
    values are initialized to zero.

    Args:
        env: A discrete-action environment instance, such as ``FrozenLakeEnv``,
            whose observation and action spaces expose an ``n`` attribute.

    Returns:
        A NumPy array of zeros with shape
        ``(env.observation_space.n, env.action_space.n)``.
    """
    observations_n = env.observation_space.n
    actions_n = env.action_space.n

    q_table = np.zeros((observations_n, actions_n))

    return q_table
