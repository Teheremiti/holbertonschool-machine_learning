#!/usr/bin/env python3
""" Neuron class """
import numpy as np


class Neuron:
    """ Defines a single neuron performing binary classification """

    def __init__(self, nx):
        """
        Constructor method for Neuron instances.

        Args:
            nx (int): The number of input features to the neuron. Must be
            positive.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be positive')
        else:
            self.__W = np.random.normal(loc=0, scale=1, size=(1, nx))
            self.__b = 0
            self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Args:
            X(np.ndarray): Matrix of size (nx, m) that contains the input data.

        Returns:
            The activated output of the neuron using sigmoid activation.
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            Y (np.ndarray): Matrix of shape (1, m) that contains the correct
            labels for the input data.
            A (np.ndarray): Matrix of shape (1, m) containing the activated
            output of the neuron for each example.

        Returns:
            The cost of the neuron.
        """
        m = Y.shape[1]
        Z = np.multiply(Y, np.log(A))\
            + np.multiply((1 - Y), np.log(1.0000001 - A))
        cost = -(1 / m) * np.sum(Z)
        return cost
