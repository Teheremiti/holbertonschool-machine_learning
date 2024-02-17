#!/usr/bin/env python3
""" forward_prop function """
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        x (_type_): _description_
        layer_sizes (list, optional): _description_. Defaults to [].
        activations (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    placeholder = x
    for i in range(len(layer_sizes)):
        prediction = create_layer(placeholder, layer_sizes[i], activations[i])
        placeholder = prediction
    return prediction
