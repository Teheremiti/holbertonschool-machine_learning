#!/usr/bin/env python3
"""Policy Gradient training."""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Implement full training with REINFORCE (Monte Carlo policy gradient).

    Args:
        env: Initial environment.
        nb_episodes: Number of episodes used for training.
        alpha: The learning rate.
        gamma: The discount factor.

    Returns:
        List of scores (sum of all rewards during one episode) per episode.
    """
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    weight = np.random.rand(obs_shape[0], n_actions)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        state = np.atleast_2d(state)
        grads = []
        rewards = []
        score = 0
        done = False

        while not done:
            action, grad = policy_gradient(state, weight)
            step_out = env.step(action)
            next_state, reward, terminated, truncated, _ = step_out
            done = terminated or truncated
            next_state = np.atleast_2d(next_state)

            score += reward
            grads.append(grad)
            rewards.append(reward)
            state = next_state

        for t in range(len(rewards)):
            G = sum(
                gamma ** k * rewards[t + k] for k in range(len(rewards) - t))
            weight += alpha * G * grads[t]

        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score))

    return scores
