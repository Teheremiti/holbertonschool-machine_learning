#!/usr/bin/env python3
"""GRU cell implementation.

Defines `GRUCell`, a gated recurrent unit with tanh hidden update and
softmax output.
"""
import numpy as np


def softmax(x):
    """Compute row-wise softmax.

    Args:
        x (np.ndarray): Logits of shape (m, o).

    Returns:
        np.ndarray: Probabilities of shape (m, o).
    """
    max_x = np.amax(x, axis=1, keepdims=True)  # Stability
    e_x = np.exp(x - max_x)
    return e_x / e_x.sum(axis=1, keepdims=True)


def sigmoid(x):
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-x))


class GRUCell:
    """Gated recurrent unit cell.

    Attributes:
        Wz (np.ndarray): Update gate weights, shape (i + h, h).
        Wr (np.ndarray): Reset gate weights, shape (i + h, h).
        Wh (np.ndarray): Candidate hidden weights, shape (i + h, h).
        Wy (np.ndarray): Output weights, shape (h, o).
        bz (np.ndarray): Update gate bias, shape (1, h).
        br (np.ndarray): Reset gate bias, shape (1, h).
        bh (np.ndarray): Candidate bias, shape (1, h).
        by (np.ndarray): Output bias, shape (1, o).
    """

    def __init__(self, i, h, o):
        """Initialize GRU cell.

        Args:
            i (int): Data dimensionality.
            h (int): Hidden state dimensionality.
            o (int): Output dimensionality.
        """
        i_h_concat = i + h
        self.Wz = np.random.normal(size=(i_h_concat, h))
        self.Wr = np.random.normal(size=(i_h_concat, h))
        self.Wh = np.random.normal(size=(i_h_concat, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros(shape=(1, h))
        self.bz = np.zeros(shape=(1, h))
        self.br = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """One GRU step forward.

        Args:
            h_prev (np.ndarray): Previous hidden state, shape (m, h).
            x_t (np.ndarray): Input at time t, shape (m, i).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - h_next: Next hidden state, shape (m, h).
                - y: Output probabilities, shape (m, o).
        """

        # Gates from concatenated [h_prev, x_t]
        hx = np.concatenate((h_prev, x_t), axis=1)
        z_t = sigmoid(hx @ self.Wz + self.bz)
        r_t = sigmoid(hx @ self.Wr + self.br)

        # Candidate hidden: use reset-applied hidden concat with x_t
        rx = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(rx @ self.Wh + self.bh)

        # Interpolate previous and candidate
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # Output
        y = softmax(h_next @ self.Wy + self.by)

        return h_next, y
