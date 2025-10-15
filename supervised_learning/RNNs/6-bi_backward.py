#!/usr/bin/env python3
"""Bidirectional RNN backward cell."""
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
    """Bidirectional RNN cell (forward and backward states)."""

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
