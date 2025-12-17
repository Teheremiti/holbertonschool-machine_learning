#!/usr/bin/env python3
"""Utilities for loading the FrozenLake environment from Gym."""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load the premade ``FrozenLakeEnv`` environment.

    This is a thin wrapper around :func:`gym.make` specialized for the
    FrozenLake environment. It forwards the provided configuration to Gym and
    returns the created environment instance.

    Args:
        desc: Optional custom map description to load instead of a premade
            layout. When provided, this is typically a list of lists of
            single-character strings describing each grid cell.
        map_name: Optional name of a premade FrozenLake map to load, such as
            ``"4x4"`` or ``"8x8"``.
        is_slippery: If ``True``, the agent may slip on the ice; if ``False``,
            transitions are deterministic.

    Returns:
        The created FrozenLake environment instance. Rendering uses the
        ``"ansi"`` mode so that calls to ``env.render()`` return strings.
    """
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi",
    )

    return env
