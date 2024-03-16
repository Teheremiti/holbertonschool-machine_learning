#!/usr/bin/env python3
"""  Forward Propagation with Dropout """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Args:
        X (ndarrray): Matrix of shape (nx, m) containing the input data for
            the network.
            - nx is the number of input features.
            - m is the number of data points.
        weights (dict): dictionary of the weights and biases of the neural
            network.
        L (): The number of layers in the network.
        keep_prob (): The probability that a node will be kept.

    Returns:
        dict: A dictionary containing the outputs of each layer and the dropout
            mask used on each layer.
    """
    cache = {'A0': X}
    for layer in range(1, L):
        Z = (np.matmul(weights['W' + str(layer)],
                       cache['A' + str(layer-1)]) + weights['b' + str(layer)])
        A = np.tanh(Z)
        dropout = np.random.binomial(1, keep_prob, size=A.shape)
        cache['D' + str(layer)] = dropout
        A = np.multiply(A, dropout) / keep_prob
        cache['A' + str(layer)] = A

    Z = (np.matmul(weights['W' + str(L)],
                   cache['A' + str(L - 1)]) + weights['b' + str(L)])
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    cache['A' + str(L)] = A

    return cache
