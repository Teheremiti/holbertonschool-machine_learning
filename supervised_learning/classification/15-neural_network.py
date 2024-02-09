#!/usr/bin/env python3
""" NeuralNetwork Class """
import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network and
        updates the private attributes __W1, __b1, __W2, and __b2.

        Args:
            X (ndarray): Matrix with shape (nx, m) containing the input data.
            Y (ndarray): Matrix with shape (1, m) that contains the correct
                labels for the input data.
            A1 (ndarray): The outpu of the hidden layer.
            A2 (ndarray): The predicted output.
            alpha (float, optional): The learning rate. Defaults to 0.05.
        """
        m = X.shape[1]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.matmul(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.matmul(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neural network.

        Args:
            X (ndarray): Matrix with shape (nx, m) that contains the input
                data.
            Y (ndarray): Matrix with shape (1, m) that contains the correct
                labels for the input data.
            iterations (int, optional): The number of iterations to train over.
                Defaults to 5000.
            alpha (float, optional): The learning rate. Defaults to 0.05.
            verbose (bool): Defines whether or not to print information about
                the training. Defaults to True.
            graph (bool): Defines whether or not to graph information about the
                training once the training has completed. Defaults to True.
            step (int): Defines the number of steps between each information
                printing.  Defaults to 100.

        Raises:
            TypeError: If iterations is not an integer or alpha is not a float.
            ValueError: If iterations or alpha is not positive.

        Returns:
            The evaluation of the training data.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")

        if not isinstance(graph, bool):
            raise TypeError("graph must be a boolean")

        costs = []
        count = []
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

            for i in range(iterations + 1):
                self.forward_prop(X)
                self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

                if i % step == 0 or i == 0 or i == iterations:
                    cost = self.cost(Y, self.__A2)
                    print(f"Cost after {i} iterations: {cost}")
                    costs.append(cost)
                    count.append(i)

            if graph:
                plt.plot(count, costs, 'b-')
                plt.xlabel('iteration')
                plt.ylabel('cost')
                plt.title('Training Cost')
                plt.show()

        return self.evaluate(X, Y)
