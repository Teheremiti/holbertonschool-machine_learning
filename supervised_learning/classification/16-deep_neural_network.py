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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            current = layers[i]
            previous = layers[i - 1]
            if i == 0:
                self.weights["W" + str(i+1)] = (np.random.randn(current, nx)
                                                * np.sqrt(2 / nx))
            else:
                self.weights["W" + str(i+1)] = (np.random.randn(current,
                                                                previous)
                                                * np.sqrt(2 / previous))

            self.weights["b" + str(i+1)] = np.zeros((current, 1))
