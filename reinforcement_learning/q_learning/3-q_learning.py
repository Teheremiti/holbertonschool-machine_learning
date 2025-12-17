#!/usr/bin/env python3
"""Q-learning training loop implementation."""
import numpy as np

epsilon_greedy = __import__("2-epsilon_greedy").epsilon_greedy


def train(
    env,
    Q,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1,
    min_epsilon=0.1,
    epsilon_decay=0.05,
):
    """Train a Q-learning agent on the provided environment.

    The function updates the supplied Q-table in place using the
    Q-learning update rule and epsilon-greedy exploration. Each episode
    is limited to ``max_steps`` steps.

    Args:
        env: Environment instance exposing the usual Gym API, including
            ``reset`` and ``step``.
        Q: NumPy array representing the Q-table to update.
        episodes: Total number of training episodes.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate in :math:`(0, 1]`.
        gamma: Discount factor in :math:`[0, 1]` for future rewards.
        epsilon: Initial exploration probability for epsilon-greedy.
        min_epsilon: Minimum value to which ``epsilon`` decays.
        epsilon_decay: Exponential decay rate applied to ``epsilon``
            after each episode.

    Returns:
        A tuple ``(Q, total_rewards)`` where:

        * ``Q`` is the updated Q-table.
        * ``total_rewards`` is a list of episode returns.
    """
    total_rewards = []

    for episode in range(episodes):
        episode_rewards = 0

        # Gymnasium reset returns (obs, info); older Gym returns obs only.
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state, _ = reset_result
        else:
            state = reset_result

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result

            if done and reward == 0:
                reward = -1

            episode_rewards += reward

            next_value = np.max(Q[next_state])
            Q[state, action] *= 1 - alpha
            Q[state, action] += alpha * (reward + gamma * next_value)

            state = next_state

            if done:
                break

        epsilon = (
            min_epsilon
            + (1 - min_epsilon) * np.exp(-epsilon_decay * episode)
        )

        total_rewards.append(episode_rewards)

    return Q, total_rewards
