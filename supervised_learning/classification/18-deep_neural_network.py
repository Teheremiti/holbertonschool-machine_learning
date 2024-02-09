#!/usr/bin/env python3
""" DeepNeuralNetwork Class"""
import numpy as np


class DeepNeuralNetwork:
    """ Defines a deep neural network performing binary classification """

    def __init__(self, nx, layers):
        """
        Class constructor function for DeepNeuralNetwork instances.

        Args:
            nx (int): The number of input features.
            layers (list): List representing the number of nodes in each layer
                of the network.

        Raises:
            TypeError: If nx is not an integer or layers is not a list of
                positive integers.
            ValueError: If nx is not positive.
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        if (not isinstance(layers, list) or
                not all(map(lambda x: isinstance(x, int) and x > 0, layers))):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            current = layers[i]
            previous = layers[i - 1]
            if i == 0:
                self.__weights["W" + str(i+1)] = (np.random.randn(current, nx)
                                                  * np.sqrt(2 / nx))
            else:
                self.__weights["W" + str(i+1)] = (np.random.randn(current,
                                                                  previous)
                                                  * np.sqrt(2 / previous))

            self.__weights["b" + str(i+1)] = np.zeros((current, 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (ndarray): Matrix with shape (nx, m) that contains the input data

        Returns:
            The output of the neural network and the cache.
        """

        if 'A0' not in self.__cache:
            self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            if i == 1:
                W = self.__weights["W{}".format(i)]
                b = self.__weights["b{}".format(i)]
                Z = np.matmul(W, X) + b
            else:
                W = self.__weights["W{}".format(i)]
                b = self.__weights["b{}".format(i)]
                X = self.__cache['A{}'.format(i-1)]
                Z = np.matmul(W, X) + b

            self.__cache["A{}".format(i)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A{}".format(i)], self.__cache
