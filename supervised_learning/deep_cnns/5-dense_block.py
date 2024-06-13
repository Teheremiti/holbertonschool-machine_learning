#!/usr/bin/env python3
""" Dense block """
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block using the bottleneck layers used for DenseNet-B.
    Each layer takes all preceding feature-maps as input.
    All weights should use the he normal initialization. The seed for the
    he_normal initializer should be set to zero.
    All convolutions should be preceded by Batch Normalization and a rectified
    linear activation (ReLU).

    Args:
        X (tensor): The output of the previous layer.
        nb_filters (int): The number of filters in X.
        growth_rate (float): The growth rate for the dense block.
        layers (int): The number of layers in the dense block.

    Returns:
        The concatenated output of each layer within the Dense Block and the
        number of filters within the concatenated outputs.
    """
    Conv2D = K.layers.Conv2D
    BatchNorm = K.layers.BatchNormalization
    Activation = K.layers.Activation

    he_normal = K.initializers.HeNormal(seed=0)

    A_prev = X
    for _ in range(layers):
        # Bottleneck (1x1) convolution with 4*k channels
        batch_norm = BatchNorm()(A_prev)
        activation = Activation('relu')(batch_norm)
        conv1x1 = Conv2D(filters=4*growth_rate,
                         kernel_size=1,
                         padding='same',
                         kernel_initializer=he_normal)(activation)

        # 3x3 convolution with k channels
        batch_norm = BatchNorm()(conv1x1)
        activation = Activation('relu')(batch_norm)
        conv3x3 = Conv2D(filters=growth_rate,
                         kernel_size=3,
                         padding='same',
                         kernel_initializer=he_normal)(activation)

        # Update parameters
        A_prev = K.layers.Concatenate()([A_prev, conv3x3])
        nb_filters += growth_rate

    output = A_prev
    return output, nb_filters
