#!/usr/bin/env python3
""" LeNet-5 with Keras """
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras.
    The model consists of the following layers in order:
        - Convolutional layer with 6 kernels of shape 5x5 with same padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Convolutional layer with 16 kernels of shape 5x5 with valid padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Fully connected layer with 120 nodes
        - Fully connected layer with 84 nodes
        - Fully connected softmax output layer with 10 nodes

    Args:
        X (K.Input): Matrix of shape (m, 28, 28, 1) containing the input images
            for the neural network.
            - m is the number of images.

    Returns:
         K.Model: The compiled model to use Adam optimization (with default
            hyperparameters) and accuracy metrics.
    """
    initializer = K.initializers.HeNormal(seed=0)

    model = K.Sequential([
        X,
        K.layers.Conv2D(filters=6,
                        kernel_size=5,
                        padding='same',
                        activation='relu',
                        kernel_initializer=initializer),
        K.layers.MaxPool2D(pool_size=2, strides=2),
        K.layers.Conv2D(filters=16,
                        kernel_size=5,
                        activation='relu',
                        kernel_initializer=initializer),
        K.layers.MaxPool2D(pool_size=2, strides=2),
        K.layers.Flatten(),
        K.layers.Dense(units=120,
                       activation='relu',
                       kernel_initializer=initializer),
        K.layers.Dense(units=84,
                       activation='relu',
                       kernel_initializer=initializer),
        K.layers.Dense(units=10,
                       activation='softmax',
                       kernel_initializer=initializer)
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])

    return model
