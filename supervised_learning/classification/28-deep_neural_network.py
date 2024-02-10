#!/usr/bin/env python3
""" DeepNeuralNetwork Class"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """ Defines a deep neural network performing binary classification """

    def __init__(self, nx, layers, activation='sig'):
        """
        Class constructor function for DeepNeuralNetwork instances.

        Args:
            nx (int): The number of input features.
            layers (list): List representing the number of nodes in each layer
                of the network.
            activation (str): Defines the type of activation function used in
                the hidden layers. Defaults to 'sig'.

        Raises:
            TypeError: If nx is not an integer or layers is not a list of
                positive integers.
            ValueError: If nx is not positive or activation is neither 'sig' or
                'tanh'.
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

        if activation not in ['sig, tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

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

    @property
    def activation(self):
        return self.__activation

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (ndarray): Matrix with shape (nx, m) that contains the input data

        Returns:
            The output of the neural network and the cache.
        """
        self.__cache['A0'] = X
        L = self.__L

        for i in range(1, L):
            Z = (np.matmul(self.__weights["W" + str(i)],
                           self.__cache['A' + str(i - 1)]) +
                 self.__weights['b' + str(i)])
            if self.__activation == 'sig':
                A = 1 / (1 + np.exp(-Z))
            else:
                A = np.tanh(Z)
            self.__cache['A' + str(i)] = A

        Z = (np.matmul(self.__weights["W" + str(L)],
                       self.__cache['A' + str(L - 1)]) +
             self.__weights['b' + str(L)])
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        self.__cache['A' + str(L)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates cost of the model using logistic regression.

        Args:
            Y (ndarray): Matrix with shape (1, m) that contains the correct
                labels for the input data.
            A (ndarray): Matrix with shape (1, m) that contains the correct
                labels for the input data.

        Returns:
            _type_: _description_
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the network's prediction.

        Args:
            X (ndarray): Matrix with shape (nx, m) that contains the input
                data.
            Y (ndarray): Matrix with shape (1, m) that contains the correct
                labels for the input data.

        Returns:
            The neuron’s prediction and the cost of the network.
        """
        output, cache = self.forward_prop(X)
        cost = self.cost(Y, output)
        predictions = np.zeros_like(output)
        predictions[np.argmax(output, axis=0), np.arange(output.shape[1])] = 1

        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on neural network.

        Args:
            Y (ndarray): Matrix with shape (1, m) that contains the correct
                labels for the input data.
            cache (dict): Contains all the intermediary value of the network.
            alpha (float): The learning rate.
        """
        L = self.__L
        m = Y.shape[1]

        dZ = cache["A{}".format(L)] - Y
        dW = np.matmul(dZ, cache["A{}".format(L - 1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        W_prev = np.copy(self.__weights["W{}".format(L)])
        self.__weights["W{}".format(L)] -= alpha * dW
        self.__weights["b{}".format(L)] -= alpha * db

        for i in range(L - 1, 0, -1):
            dA = np.matmul(W_prev.T, dZ)
            A = cache["A{}".format(i)]
            if self.__activation == 'sig':
                dZ = dA * A * (1 - A)
            else:
                dZ = dA * (1 - (A ** 2))
            dW = np.matmul(dZ, cache["A{}".format(i)].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            W_prev = np.copy(self.__weights["W{}".format(i)])
            self.__weights["W{}".format(i)] -= alpha * dW
            self.__weights["b{}".format(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network.

        Args:
            X (ndarray): Matrix with shape (nx, m) that contains the input data
            Y (ndarray): Matrix with shape (1, m) that contains the correct
                labels for the input data.
            iterations (int): The number of iterations to train over.
            alpha (float): The learning rate.
            verbose (bool): Defines whether or not to print information about
                the training.
            graph (bool): Defines whether or not to graph information about the
                training once the training has completed
            step (int): Defines the number of steps between each information
                printing. Defaults to 100.

        Returns:
            The evaluation of the training data after iterations.
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")

        if not isinstance(graph, bool):
            raise TypeError("graph must be a boolean")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        count = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)

            if i != iterations:
                self.gradient_descent(Y, cache, alpha)

            cost = self.cost(Y, A)
            costs.append(cost)
            count.append(i)

            if verbose and (i % step == 0 or i == 0 or i == iterations):
                print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(count, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves instance object to a file in pickle format.

        Args:
            filename (str): File to which the object should be saved.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object.

        filename (str): File from which object should be loaded.

        Returns:
            The loaded object or None if filename doesn't exist.
        """
        try:
            with open(filename, 'rb') as f:
                loaded_object = pickle.load(f)
            return loaded_object

        except FileNotFoundError:
            return None
