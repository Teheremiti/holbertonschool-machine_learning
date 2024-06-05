#!/usr/bin/env python3
""" LeNet-5 with Tensorflow 1 """
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow.
    The model consists of the following layers in order:
        - Convolutional layer with 6 kernels of shape 5x5 with same padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Convolutional layer with 16 kernels of shape 5x5 with valid padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Fully connected layer with 120 nodes
        - Fully connected layer with 84 nodes
        - Fully connected softmax output layer with 10 nodes

    Args:
        x (tf.placeholder): Matrix of shape (m, 28, 28, 1) containing the
            input images for the network.
            - m is the number of images
        y (tf.placeholder): Matrix of shape (m, 10) containing the one-hot
            labels for the network.

    Returns:
        - a tensor for the softmax activated output
        - a training operation that utilizes Adam optimization (with default
          hyperparameters)
        - a tensor for the loss of the netowrk
        - a tensor for the accuracy of the network
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv1 = tf.layers.Conv2D(filters=6,
                             kernel_size=5,
                             padding='same',
                             activation='relu',
                             kernel_initializer=initializer)(x)

    max_pool1 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv1)

    conv2 = tf.layers.Conv2D(filters=16,
                             kernel_size=5,
                             padding='valid',
                             activation='relu',
                             kernel_initializer=initializer)(max_pool1)

    max_pool2 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)

    flatten = tf.layers.Flatten()(max_pool2)

    dense1 = tf.layers.Dense(units=120,
                             activation='relu',
                             kernel_initializer=initializer)(flatten)

    dense2 = tf.layers.Dense(units=84,
                             activation='relu',
                             kernel_initializer=initializer)(dense1)

    output = tf.layers.Dense(units=10,
                             kernel_initializer=initializer)(dense2)

    activated_output = tf.nn.softmax(output)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)

    adam_optimizer = tf.train.AdamOptimizer().minimize(loss)

    y_pred = tf.argmax(output, axis=1)
    y_true = tf.argmax(y, axis=1)
    correct_prediction = tf.cast(tf.equal(y_pred, y_true), dtype=tf.float32)

    accuracy = tf.reduce_mean(correct_prediction)

    return activated_output, adam_optimizer, loss, accuracy
