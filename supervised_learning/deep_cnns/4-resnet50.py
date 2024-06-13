#!/usr/bin/env python3
""" ResNet-50 """
from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture.
    The input data is assumed to have shape (224, 224, 3).
    All convolutions inside and outside the blocks should be followed by batch
    normalization along the channels axis and a ReLU activation.
    All weights should use he normal initialization. The seed for the he_normal
    initializer should be set to zero.

    Returns:
        K.Model: The keras model.
    """
    X = K.Input((224, 224, 3))
    he_normal = K.initializers.HeNormal(seed=0)

    # Pre-processing : 7x7 convolution with stride 2
    conv0 = K.layers.Conv2D(filters=64,
                            kernel_size=7,
                            strides=2,
                            padding='same',
                            kernel_initializer=he_normal)(X)
    batch_norm0 = K.layers.BatchNormalization()(conv0)
    output0 = K.layers.Activation('relu')(batch_norm0)

    # First layer : max_pool3x3/2, projection block s=1, two identity blocks
    filters = (64, 64, 256)
    max_pool0 = K.layers.MaxPooling2D(pool_size=3,
                                      strides=2,
                                      padding='same')(output0)
    proj1 = projection_block(max_pool0, filters, s=1)
    id1 = identity_block(proj1, filters)
    output1 = identity_block(id1, filters)

    # Second layer : Projection block s=2, three identity blocks
    filters = (128, 128, 512)
    proj2 = projection_block(output1, filters)
    id2a = identity_block(proj2, filters)
    id2b = identity_block(id2a, filters)
    output2 = identity_block(id2b, filters)

    # Third layer : Projection block s=2, five identity blocks
    filters = (256, 256, 1024)
    proj3 = projection_block(output2, filters)
    id3a = identity_block(proj3, filters)
    id3b = identity_block(id3a, filters)
    id3c = identity_block(id3b, filters)
    id3d = identity_block(id3c, filters)
    output3 = identity_block(id3d, filters)

    # Fourth layer : Projection block s=2, two identity blocks
    filters = (512, 512, 2048)
    proj4 = projection_block(output3, filters)
    id4 = identity_block(proj4, filters)
    output4 = identity_block(id4, filters)

    # Dense fully connected layer
    avg = K.layers.AveragePooling2D(pool_size=7)(output4)
    dense = K.layers.Dense(units=1000, activation='softmax')(avg)

    # Initialize model
    model = K.Model(X, dense)

    return model
