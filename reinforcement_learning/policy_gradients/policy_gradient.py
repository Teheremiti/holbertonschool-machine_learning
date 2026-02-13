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
