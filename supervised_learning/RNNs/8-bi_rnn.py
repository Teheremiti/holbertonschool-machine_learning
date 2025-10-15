#!/usr/bin/env python3
"""Bidirectional RNN forward propagation."""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Forward propagation for a bidirectional RNN.

    Args:
        bi_cell: Instance of `BidirectionalCell`.
        X (np.ndarray): Input data, shape (t, m, i).
        h_0 (np.ndarray): Initial forward hidden state, shape (m, h).
        h_t (np.ndarray): Initial backward hidden state, shape (m, h).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - H: Concatenated hidden states, shape (t, m, 2*h).
            - Y: Outputs, shape (t, m, o).

    Raises:
        TypeError: If inputs are not numpy arrays.
        ValueError: If shapes are incompatible.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError('X should be a ndarray')
    if not isinstance(h_0, np.ndarray):
        raise TypeError('h_0 should be a ndarray')
    if not isinstance(h_t, np.ndarray):
        raise TypeError('h_t should be a ndarray')
    if X.shape[1] != h_0.shape[0] or X.shape[1] != h_t.shape[0]:
        raise ValueError('Verify your dimension')

    t, m, _ = X.shape
    _, h = h_0.shape

    H_f = np.zeros((t + 1, m, h))
    H_b = np.zeros((t + 1, m, h))
    H_f[0] = h_0
    H_b[-1] = h_t

    for t_idx in range(t):
        H_f[t_idx + 1] = bi_cell.forward(H_f[t_idx], X[t_idx])

    for t_idx in range(t - 1, -1, -1):
        H_b[t_idx] = bi_cell.backward(H_b[t_idx + 1], X[t_idx])

    H = np.concatenate((H_f[1:], H_b[:-1]), axis=2)
    Y = bi_cell.output(H)
    return H, Y
