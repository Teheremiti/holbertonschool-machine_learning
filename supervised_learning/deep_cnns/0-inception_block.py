#!/usr/bin/env python3
""" Inception Block """
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block. All convolutions inside the inception block
    use a rectified linear activation (ReLU).

    Args:
        A_prev (K.Input): The output from the previous layer.
        filters ({tuple, list}): Contains F1, F3R, F3,F5R, F5, FPP:
            - F1 is the number of filters in the 1x1 convolution
            - F3R is the number of filters in the 1x1 convolution before
              the 3x3 convolution
            - F3 is the number of filters in the 3x3 convolution
            - F5R is the number of filters in the 1x1 convolution before
              the 5x5 convolution
            - F5 is the number of filters in the 5x5 convolution
            - FPP is the number of filters in the 1x1 convolution after
              the max pooling

    Returns:
        K.Output: The concatenated output of the inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution branch
    conv1x1 = K.layers.Conv2D(F1,
                              (1, 1),
                              padding='same',
                              activation='relu')(A_prev)

    # 1x1 convolution followed by 3x3 convolution branch
    conv3x3_reduce = K.layers.Conv2D(F3R,
                                     (1, 1),
                                     padding='same',
                                     activation='relu')(A_prev)
    conv3x3 = K.layers.Conv2D(F3,
                              (3, 3),
                              padding='same',
                              activation='relu')(conv3x3_reduce)

    # 1x1 convolution followed by 5x5 convolution branch
    conv5x5_reduce = K.layers.Conv2D(F5R,
                                     (1, 1),
                                     padding='same',
                                     activation='relu')(A_prev)
    conv5x5 = K.layers.Conv2D(F5,
                              (5, 5),
                              padding='same',
                              activation='relu')(conv5x5_reduce)

    # Max pooling followed by 1x1 convolution branch
    max_pool = K.layers.MaxPooling2D((3, 3),
                                     strides=(1, 1),
                                     padding='same')(A_prev)
    conv_pool_proj = K.layers.Conv2D(FPP,
                                     (1, 1),
                                     padding='same',
                                     activation='relu')(max_pool)

    # Concatenate all the branches
    output = K.layers.concatenate([conv1x1, conv3x3, conv5x5, conv_pool_proj],
                                  axis=-1)

    return output
