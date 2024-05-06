#!/usr/bin/env python3
""" Input """

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.

    Args:
        nx (int): The number of input features to the network.
        layers (list): Contains the number of nodes in each layer of the
            network.
        activations (list): Contains the activation functions used for each
            layer of the network.
        lambtha (float): The L2 regularization parameter.
        keep_prob (float): Probability that a node will be kept for dropout.

    Returns:
        The Keras model.
    """
    inputs = K.Input(shape=(nx,))
    x = inputs
    for i in range(len(layers)):
        x = K.layers.Dense(layers[i],
                           activation=activations[i],
                           kernel_regularizer=K.regularizers.L2(lambtha))(x)

        if i != len(layers) - 1 and keep_prob is not None:
            x = K.layers.Dropout(1 - keep_prob)(x)
    model = K.Model(inputs, x)
    return model
