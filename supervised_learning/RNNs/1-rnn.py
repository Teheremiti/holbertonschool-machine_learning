#!/usr/bin/env python3
"""Simple RNN forward propagation."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Forward propagation for a simple RNN.

    Args:
        rnn_cell: Instance of `RNNCell`.
        X (np.ndarray): Input data of shape (t, m, i).
        h_0 (np.ndarray): Initial hidden state of shape (m, h).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - H: All hidden states, shape (t + 1, m, h), with H[0] as zeros.
            - Y: All outputs, shape (t, m, o).

    Raises:
        TypeError: If X or h_0 are not numpy arrays.
        ValueError: If X and h_0 have incompatible shapes.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError('X should be a ndarray')
    if not isinstance(h_0, np.ndarray):
        raise TypeError('h_0 should be a ndarray')
    if X.shape[1] != h_0.shape[0]:
        raise ValueError("X.shape is (t,m,i) and h_0 shape is (m,h)")

    # Shapes
    t, m, i = X.shape
    _, h = h_0.shape
    o = rnn_cell.by.shape[1]

    # Outputs; H[0] remains zeros to match existing behavior
    h_prev = h_0
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    # Time steps
    for t_idx in range(t):
        h_next, y = rnn_cell.forward(h_prev, X[t_idx])
        H[t_idx + 1] = h_next
        Y[t_idx] = y
        h_prev = h_next

    return H, Y
