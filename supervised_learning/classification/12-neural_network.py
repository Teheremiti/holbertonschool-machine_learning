#!/usr/bin/env python3
""" NeuralNetwork Class """
import numpy as np


class NeuralNetwork:
    """ Defines a neural network with one hidden layer performing
    binary classification. """

    def __init__(self, nx, nodes):
        """
        Class constructor function for NeuralNetwork instances.

        Args:
            nx (int): The number of input features.
            nodes (int): The number of nodes found in the hidden layer.

        Raises:
            TypeError: If nx or nodes is not an integer.
            ValueError: If nx or nodes is not strictly positive.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        elif nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (np.ndarray): Matrix with shape (nx, m) that contains the input
            data.

        Returns:

        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        Args:
            Y (ndarray): Matrix with shape (1, m) that contains the correct
                labels for the input data.
            A (ndarray): Matrix with shape (1, m) containing the activated
                output of the neuron for each example.

        Returns:
            The cost of the model.
        """
        m = Y.shape[1]
        cost = -(1/m) * np.sum((Y * np.log(A) + (1-Y) * np.log(1.0000001-A)))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions.

        Args:
            X (ndarray): Matrix with shape (nx, m) containing the input data.
            Y (ndarray): Matrix with shape (1, m) that contains the correct
                labels for the input data.

        Returns:
            The neuron’s prediction and the cost of the network.
        """
        self.forward_prop(X)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)

        return prediction, cost
