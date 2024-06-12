#!/usr/bin/env python3
""" Projection Block """
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block. All convolutions inside the block should be
    followed by batch normalization along the channels axis and a rectified
    linear activation (ReLU), respectively.
    All weights should use he normal initialization. The seed for the he_normal
    initializer should be set to zero.

    Args:
        A_prev (K.Input): The output from the previous layer.
        filters ({tuple, list}): Contains F11, F3, F12, respectively:
            - F11 is the number of filters in the first 1x1 convolution
            - F3 is the number of filters in the 3x3 convolution
            - F12 is the number of filters in the second 1x1 convolution as
              well as the 1x1 convolution in the shortcut connection
        s (int, optional): The stride of the first convolution in both the main
            path and the shortcut connection. Defaults to 2.

    Returns:
        The activated output of the projection block.
    """
    F11, F3, F12 = filters
    he_normal = K.initializers.HeNormal(seed=0)

    # First layer left branch with stride s
    conv1x1a = K.layers.Conv2D(filters=F11,
                               kernel_size=1,
                               strides=s,
                               padding='same',
                               kernel_initializer=he_normal)(A_prev)
    batch_norm1 = K.layers.BatchNormalization()(conv1x1a)
    activation1 = K.layers.Activation(activation='relu')(batch_norm1)

    # Second layer left branch
    conv3x3 = K.layers.Conv2D(filters=F3,
                              kernel_size=3,
                              padding='same',
                              kernel_initializer=he_normal)(activation1)
    batch_norm2 = K.layers.BatchNormalization()(conv3x3)
    activation2 = K.layers.Activation(activation='relu')(batch_norm2)

    # Third layer left branch
    conv1x1b = K.layers.Conv2D(filters=F12,
                               kernel_size=1,
                               padding='same',
                               kernel_initializer=he_normal)(activation2)
    batch_norm3 = K.layers.BatchNormalization()(conv1x1b)

    # Right branch layer with stride s
    conv1x1R = K.layers.Conv2D(filters=F12,
                               kernel_size=1,
                               strides=s,
                               padding='same',
                               kernel_initializer=he_normal)(A_prev)
    batch_normR = K.layers.BatchNormalization()(conv1x1R)

    # Output
    add = K.layers.Add()([batch_norm3, batch_normR])
    output = K.layers.Activation(activation='relu')(add)

    return output
