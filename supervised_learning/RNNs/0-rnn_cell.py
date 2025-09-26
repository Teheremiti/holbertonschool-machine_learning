#!/usr/bin/env python3
"""
Simple RNN cell.

This module defines `RNNCell`, a single recurrent cell using tanh for the
hidden state and softmax for the output.
"""
import numpy as np


def softmax(x):
    """
    Compute row-wise softmax.

    Args:
        x (np.ndarray): Logits of shape (m, o).

    Returns:
        np.ndarray: Probabilities of shape (m, o) that sum to 1 per row.
    """
    # Subtract row-wise max for numerical stability
    max_x = np.amax(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_x)
    return e_x / e_x.sum(axis=1, keepdims=True)


class RNNCell:
    """
    A cell of a simple RNN.

    Attributes:
        Wh (np.ndarray): Weights for concatenated input and hidden state,
            shape (i + h, h), initialized from N(0, 1).
        Wy (np.ndarray): Weights for output, shape (h, o), initialized from
            N(0, 1).
        bh (np.ndarray): Bias for hidden state, shape (1, h), initialized to 0.
        by (np.ndarray): Bias for output, shape (1, o), initialized to 0.
    """

    def __init__(self, i, h, o):
        """
        Initialize the RNN cell.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        i_h_concat = i + h
        self.Wh = np.random.normal(size=(i_h_concat, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """
        Forward propagation for one time step.

        Args:
            h_prev (np.ndarray): Previous hidden state of shape (m, h).
            x_t (np.ndarray): Input at time t of shape (m, i).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - h_next: Next hidden state of shape (m, h) after tanh.
                - y: Output probabilities of shape (m, o) after softmax.
        """

        # Concatenate hidden state and input
        x_concat = np.concatenate((h_prev, x_t), axis=1)

        # Next hidden state (tanh)
        h_next = np.tanh((x_concat @ self.Wh + self.bh))

        # Output probabilities (softmax)
        y = softmax(h_next @ self.Wy + self.by)

        return h_next, y
