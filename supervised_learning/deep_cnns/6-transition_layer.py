#!/usr/bin/env python3
""" Transition layer """
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer by implementing compression as used in
    DenseNet-C.
    All weights should use he normal initialization. The seed for the he_normal
    initializer should be set to zero.
    All convolutions should be preceded by Batch Normalization and a rectified
    linear activation (ReLU).

    Args:
        X (tensor): The output of the previous layer.
        nb_filters (int): The number of filters in the input (X).
        compression (float): The compression factor for the transition layer.

    Returns:
        The output of the transition layer and the number of filters within.
    """
    he_normal = K.initializers.HeNormal(seed=0)

    # Compressed nb of filters
    filters = int(nb_filters * compression)


    # Transition layer : Bottleneck convolution compressed by the θ factor,
    # followed by a 2x2/2 average pooling.
    batch_norm = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch_norm)
    conv = K.layers.Conv2D(filters=filters,
                           kernel_size=1,
                           padding='same',
                           kernel_initializer=he_normal)(activation)
    avg_pool = K.layers.AveragePooling2D(pool_size=2, strides=2)(conv)

    return avg_pool, filters
