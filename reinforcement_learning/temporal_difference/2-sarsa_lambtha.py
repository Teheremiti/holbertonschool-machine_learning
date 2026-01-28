#!/usr/bin/env python3
"""SARSA(λ) algorithm for action-value function estimation."""
import numpy as np


def epsilon_greedy(state, Q, epsilon):
    """
    Select an action using epsilon-greedy policy.

    With probability epsilon, selects a random action (exploration).
    Otherwise, selects the action with highest Q-value (exploitation).

    Args:
        state: Current state of the environment.
        Q: numpy.ndarray of shape (s, a) containing the Q table.
        epsilon: Exploration rate threshold.

    Returns:
        int: Selected action index.
    """
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(Q[state, :])
    return np.random.randint(0, int(Q.shape[1]))


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Perform the SARSA(λ) algorithm for action-value estimation.

    Uses eligibility traces with the SARSA update rule to learn an
    action-value function while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment instance.
        Q: numpy.ndarray of shape (s, a) containing the Q table.
        lambtha: Eligibility trace decay factor (0 to 1).
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate for Q-value updates.
        gamma: Discount rate for future rewards.
        epsilon: Initial exploration rate for epsilon-greedy.
        min_epsilon: Minimum value epsilon should decay to.
        epsilon_decay: Decay rate for epsilon between episodes.

    Returns:
        numpy.ndarray: Updated Q table.
    """
    epsilon_init = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(state, Q, epsilon)
        eligibility = np.zeros_like(Q)

        for step in range(max_steps):
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            next_action = epsilon_greedy(next_state, Q, epsilon)

            delta = (reward + gamma * Q[next_state, next_action]
                     - Q[state, action])

            eligibility[state, action] += 1
            eligibility *= lambtha * gamma

            Q += alpha * delta * eligibility

            if done:
                break

            state = next_state
            action = next_action

        epsilon = (min_epsilon + (epsilon_init - min_epsilon)
                   * np.exp(-epsilon_decay * episode))

    return Q
