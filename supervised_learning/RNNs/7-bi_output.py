#!/usr/bin/env python3
"""Bidirectional RNN outputs."""
import numpy as np


def softmax(x):
    """Compute row-wise softmax."""
    max_x = np.amax(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_x)
    return e_x / e_x.sum(axis=1, keepdims=True)


def sigmoid(x):
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-x))


class BidirectionalCell:
    """Bidirectional RNN cell with output computation."""

    def __init__(self, i, h, o):
        """Initialize bidirectional cell.

        Args:
            i (int): Data dimensionality.
            h (int): Hidden state dimensionality.
            o (int): Output dimensionality.
        """
        i_h_concat = i + h
        self.Whf = np.random.normal(size=(i_h_concat, h))
        self.Whb = np.random.normal(size=(i_h_concat, h))
        self.Wy = np.random.normal(size=(h * 2, o))
        self.bhf = np.zeros(shape=(1, h))
        self.bhb = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """Forward-direction hidden state for one time step.

        Args:
            h_prev (np.ndarray): Previous hidden state, shape (m, h).
            x_t (np.ndarray): Input at time t, shape (m, i).

        Returns:
            np.ndarray: Next forward hidden state, shape (m, h).
        """
        xh = np.concatenate((h_prev, x_t), axis=1)
        return np.tanh(xh @ self.Whf + self.bhf)

    def backward(self, h_next, x_t):
        """Backward-direction hidden state for one time step.

        Args:
            h_next (np.ndarray): Next hidden state, shape (m, h).
            x_t (np.ndarray): Input at time t, shape (m, i).

        Returns:
            np.ndarray: Previous backward hidden state, shape (m, h).
        """
        xh = np.concatenate((h_next, x_t), axis=1)
        return np.tanh(xh @ self.Whb + self.bhb)

    def output(self, H):
        """Compute outputs for all time steps.

        Args:
            H (np.ndarray): Concatenated hidden states, shape (t, m, 2*h).

        Returns:
            np.ndarray: Outputs, shape (t, m, o).
        """
        t, m, _ = H.shape
        Y = np.empty((t, m, self.Wy.shape[1]))
        for t_idx in range(t):
            logits = H[t_idx] @ self.Wy + self.by
            Y[t_idx] = softmax(logits)
        return Y
