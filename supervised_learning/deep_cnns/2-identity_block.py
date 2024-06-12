#!/usr/bin/env python3
""" Identity Block """

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block.
    All convolutions inside the block are followed by batch normalization
    along the channels axis and a rectified linear activation (ReLU).
    All weights use the he normal initialization. The seed for the he_normal
    initializer should be set to zero.

    Args:
        A_prev (K.Input): The output of the previous layer.
        filters ({tuple, list}): Contains F11, F3, F12, respectively:
            - F11 is the number of filters in the first 1x1 convolution
            - F3 is the number of filters in the 3x3 convolution
            - F12 is the number of filters in the second 1x1 convolution

    Returns:
        The activated output of the identity block.
    """
    F11, F3, F12 = filters

    # Initializer
    he_normal = K.initializers.HeNormal(seed=0)

    # First convolution followed by batch normalization and ReLU activation
    conv1x1a = K.layers.Conv2D(filters=F11,
                               kernel_size=1,
                               padding="same",
                               kernel_initializer=he_normal)(A_prev)
    batch_norm1 = K.layers.BatchNormalization()(conv1x1a)
    activation1 = K.layers.Activation(activation="relu")(batch_norm1)

    # Second convolution followed by batch normalization and ReLU activation
    conv3x3 = K.layers.Conv2D(filters=F3,
                              kernel_size=3,
                              padding="same",
                              kernel_initializer=he_normal)(activation1)
    batch_norm2 = K.layers.BatchNormalization()(conv3x3)
    activation2 = K.layers.Activation(activation="relu")(batch_norm2)

    # Third convolution followed by batch normalization, add, and ReLU
    conv1x1b = K.layers.Conv2D(filters=F12,
                               kernel_size=1,
                               padding="same",
                               kernel_initializer=he_normal)(activation2)
    batch_norm3 = K.layers.BatchNormalization()(conv1x1b)
    add = K.layers.Add()([batch_norm3, A_prev])
    output = K.layers.Activation(activation="relu")(add)

    return output
