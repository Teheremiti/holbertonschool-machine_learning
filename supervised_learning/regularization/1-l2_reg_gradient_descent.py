#!/usr/bin/env python3
""" L2 regularization - Gradient descent """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization. The neural network uses tanh activations on each
    layer except the last, which uses a softmax activation.

    Args:
        Y (ndarray): One-hot ndarray of shape (classes, m) that contains the
            correct labels for the data.
            - classes is the number of classes.
            - m is the number of data points.
        weights (dict): Dictionary of the weights and biases of the neural
            network.
        cache (dict): Dictionary of the outputs of each layer of the neural
            network.
        alpha (float): The learning rate.
        lambtha (float)): The L2 regularization parameter.
        L (int): The number of layers of the network.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        L2_regularization = lambtha / m * weights['W' + str(layer)]

        A_prev = cache['A' + str(layer - 1)]

        dW = np.matmul(dZ, A_prev.T) / m + L2_regularization
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.matmul(weights['W' + str(layer)].T, dZ)

        if layer != 1:
            dZ = dA * (1 - A_prev ** 2)
        else:
            dZ = dA

        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db
