#!/usr/bin/env python3
"""Utilities for running a trained Q-learning agent."""
import numpy as np


def play(env, Q, max_steps=100):
    """Run a single episode using a trained Q-table.

    The agent acts greedily with respect to the Q-table: at each step it
    selects the action with the highest Q-value for the current state.
    The environment is rendered at every step.

    Args:
        env: ``FrozenLakeEnv`` instance or compatible environment with
            ``reset``, ``step`` and ``render`` methods.
        Q: NumPy array containing the trained Q-table.
        max_steps: Maximum number of steps to execute in the episode.

    Returns:
        A tuple ``(total_rewards, rendered_outputs)`` where:

        * ``total_rewards`` is the cumulative reward for the episode.
        * ``rendered_outputs`` is a list of strings showing the board
          after the initial reset and after each action, with the agent
          position highlighted using backticks.
    """
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        state, _ = reset_result
    else:
        state = reset_result

    desc = env.unwrapped.desc

    def format_board(desc, state_index):
        rows, cols = desc.shape
        row, col = divmod(state_index, cols)
        lines = []
        for i in range(rows):
            chars = []
            for j in range(cols):
                cell = desc[i][j]
                if isinstance(cell, bytes):
                    cell = cell.decode("utf-8")
                if i == row and j == col:
                    chars.append(f"`{cell}`")
                else:
                    chars.append(cell)
            lines.append("".join(chars))
        return "\n".join(lines)

    action_labels = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

    rendered_outputs = [format_board(desc, state)]
    total_rewards = 0

    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        label = action_labels.get(action, str(action))

        step_result = env.step(action)
        if len(step_result) == 5:
            new_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            new_state, reward, done, info = step_result

        if isinstance(new_state, tuple):
            new_state = new_state[0]

        total_rewards += reward
        state = new_state

        rendered_outputs.append(f"  ({label})")
        rendered_outputs.append(format_board(desc, state))

        if done:
            break

    return total_rewards, rendered_outputs
