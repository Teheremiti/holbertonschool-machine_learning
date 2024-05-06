#!/usr/bin/env python3

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
        keep_prob (float): The probability that a node will be kept for dropout.

    Returns:
        The Keras model.
    """
    model = K.models.Sequential()
    for i in range(len(layers)):
        model.add(K.layers.Dense(units=layers[i],
                        activation=activations[i],
                        kernel_regularizer=K.regularizers.L2(lambtha),
                        input_dim=nx))

        if i != len(layers) - 1 and keep_prob is not None:
            model.add(K.layers.Dropout(rate=1 - keep_prob))
    return model
