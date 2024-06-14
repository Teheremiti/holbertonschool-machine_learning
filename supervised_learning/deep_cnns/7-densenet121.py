#!/usr/bin/env python3
""" DenseNet-121 """
from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture.
    The input data is assumed to have shape (224, 224, 3).
    All convolutions should be preceded by Batch Normalization and a rectified
    linear activation (ReLU).
    All weights should use he normal initialization. The seed for the he_normal
    initializer should be set to zero.

    Args:
        growth_rate (int, optional): The growth rate. Defaults to 1.0.
        compression (float, optional): The compression factor. Defaults to 32.

    Returns:
        K.Model: The keras model.
    """
    X = K.Input((224, 224, 3))
    he_normal = K.initializers.HeNormal(seed=0)

    # Pre-processing: conv7x7/2, max_pool3x3/2
    batch_norm = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch_norm)
    conv = K.layers.Conv2D(filters=64,
                           kernel_size=7,
                           strides=2,
                           padding='same',
                           kernel_initializer=he_normal)(activation)
    max_pool = K.layers.MaxPool2D(pool_size=3,
                                  strides=2,
                                  padding='same')(conv)

    # Dense block 1 with 6 layers, followed by Transition layer
    dense1, nb_filters = dense_block(max_pool, 64, growth_rate, 6)
    trans1, nb_filters = transition_layer(dense1, nb_filters, compression)

    # Dense block 2 with 12 layers, followed by Transition layer
    dense2, nb_filters = dense_block(trans1, nb_filters, growth_rate, 12)
    trans2, nb_filters = transition_layer(dense2, nb_filters, compression)

    # Dense block 3 with 24 layers, followed by Transition layer
    dense3, nb_filters = dense_block(trans2, nb_filters, growth_rate, 24)
    trans3, nb_filters = transition_layer(dense3, nb_filters, compression)

    # Dense block 4 with 16 layers, no Transition layer
    dense4, nb_filters = dense_block(trans3, nb_filters, growth_rate, 16)

    # Classification layer: Global avg_pool7x7, 1000D fully connected, softmax
    avg_pool = K.layers.AveragePooling2D(pool_size=7)(dense4)
    fc = K.layers.Dense(units=1000, activation='softmax')(avg_pool)

    # Initialize model
    model = K.Model(X, fc)

    return model
