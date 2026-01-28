#!/usr/bin/env python3
"""Monte Carlo algorithm for value function estimation."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Perform the Monte Carlo algorithm for value estimation.

    Uses first-visit Monte Carlo to estimate the value function by
    averaging returns observed after first visits to each state.

    Args:
        env: OpenAI environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimate.
        policy: Function that takes a state and returns the next action.
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate for value updates.
        gamma: Discount rate for future rewards.

    Returns:
        numpy.ndarray: Updated value estimate rounded to 4 decimal places.
    """
    for ep in range(episodes):
        state = env.reset()
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_data.append((state, reward))

            if done or step > max_steps:
                break

            state = next_state

        episode_data = np.array(episode_data, dtype=int)

        G = 0
        for s, r in episode_data[::-1]:
            G = gamma * G + r
            if s not in episode_data[:ep, 0]:
                V[s] = V[s] + alpha * (G - V[s])

    np.set_printoptions(precision=4, suppress=True)

    return V.round(4)
