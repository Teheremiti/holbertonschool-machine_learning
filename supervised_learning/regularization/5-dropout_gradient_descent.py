#!/usr/bin/env python3
""" Gradient descent with Dropout """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization using
    gradient descent. All layers use thetanh activation function except the
    last, which uses the softmax activation function.

    Args:
        Y (ndarray): One-hot ndarray of shape (classes, m) that contains the
            correct labels for the data.
            - classes is the number of classes
            - m is the number of data points.
        weights (dict): Dictionary of the weights and biases of the neural
            network.
        cache (dict): Dictionary of the outputs and dropout masks of each layer
            of the neural network.
        alpha (float): The learning rate.
        keep_prob (float): The probability that a node will be kept.
        L (int): The number of layers of the network.
    """
    m = Y.shape[1]

    A = cache['A' + str(L)]
    dZ = A - Y

    A_prev = cache['A' + str(L - 1)]
    W = weights['W' + str(L)]
    dW = np.matmul(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.matmul(W.T, dZ)

    weights['W' + str(L)] -= alpha * dW
    weights['b' + str(L)] -= alpha * db

    for layer in range(L - 1, 0, -1):
        D = cache['D' + str(layer)]
        dA = dA_prev * (D / keep_prob)

        A = cache['A' + str(layer)]
        A_prev = cache['A' + str(layer - 1)]
        dZ = dA * (1 - A ** 2)
        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        W = weights['W' + str(layer)]
        dA_prev = np.matmul(W.T, dZ)

        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db
