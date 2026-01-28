#!/usr/bin/env python3
"""TD(λ) algorithm for value function estimation."""
import numpy as np


def td_lambtha(env, V, policy, lambtha,
               episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Perform the TD(λ) algorithm for value estimation.

    Uses eligibility traces to combine benefits of TD(0) and Monte Carlo
    methods, providing a smooth interpolation between the two approaches.

    Args:
        env: OpenAI environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimate.
        policy: Function that takes a state and returns the next action.
        lambtha: Eligibility trace decay factor (0 to 1).
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate for value updates.
        gamma: Discount rate for future rewards.

    Returns:
        numpy.ndarray: Updated value estimate.
    """
    for episode in range(episodes):
        state, _ = env.reset()
        eligibility = np.zeros_like(V)

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            delta = reward + gamma * V[next_state] - V[state]

            eligibility *= lambtha * gamma
            eligibility[state] += 1

            V = V + alpha * delta * eligibility

            if done:
                break

            state = next_state

    return V
