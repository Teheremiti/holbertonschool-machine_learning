#!/usr/bin/env python3
""" Inception Network """
from tensorflow import keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds an inception network. All convolutions inside and outside the
    inception block use a rectified linear activation (ReLU). The input data
    is assumed to have a shape of (224, 224, 3).

    Returns:
        K.Model: The keras model.
    """
    # Input of the network
    X = K.Input(shape=(224, 224, 3))

    # First convolution and pooling layers
    conv7x7 = K.layers.Conv2D(filters=64,
                              kernel_size=7,
                              strides=2,
                              padding='same',
                              activation='relu')(X)
    max_pool1 = K.layers.MaxPool2D(pool_size=3,
                                   strides=2,
                                   padding='same')(conv7x7)
    conv1x1 = K.layers.Conv2D(filters=64,
                              kernel_size=1,
                              padding='same',
                              activation='relu')(max_pool1)
    conv3x3 = K.layers.Conv2D(filters=192,
                              kernel_size=3,
                              padding='same',
                              activation='relu')(conv1x1)
    max_pool2 = K.layers.MaxPool2D(pool_size=3,
                                   strides=2,
                                   padding='same')(conv3x3)

    # Inception block n°3 followed by 3X3/2 max pooling
    inception_3a = inception_block(max_pool2, (64, 96, 128, 16, 32, 32))
    inception_3b = inception_block(inception_3a, (128, 128, 192, 32, 96, 64))

    max_pool3 = K.layers.MaxPool2D(pool_size=3,
                                   strides=2,
                                   padding='same')(inception_3b)

    # Inception block n°4 followed by 3x3/2 max pooling
    inception_4a = inception_block(max_pool3, (192, 96, 208, 16, 48, 64))
    inception_4b = inception_block(inception_4a, (160, 112, 224, 24, 64, 64))
    inception_4c = inception_block(inception_4b, (128, 128, 256, 24, 64, 64))
    inception_4d = inception_block(inception_4c, (112, 144, 288, 32, 64, 64))
    inception_4e = inception_block(inception_4d, (256, 160, 320, 32, 128, 128))

    max_pool4 = K.layers.MaxPool2D(pool_size=3,
                                   strides=2,
                                   padding='same')(inception_4e)

    # Inception block n°5 followed by 7x7/1 average pooling
    inception_5a = inception_block(max_pool4, (256, 160, 320, 32, 128, 128))
    inception_5b = inception_block(inception_5a, (384, 192, 384, 48, 128, 128))

    avg_pool = K.layers.AveragePooling2D(pool_size=7)(inception_5b)

    # Regularization and activations
    dropout = K.layers.Dropout(rate=0.4)(avg_pool)

    linear = K.layers.Flatten()(dropout)

    outputs = K.layers.Dense(units=1000, activation='softmax')(linear)

    # Build model
    model = K.Model(X, outputs)

    return model
