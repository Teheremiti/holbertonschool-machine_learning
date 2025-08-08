#!/usr/bin/env python3
"""
This module defines a convolutional autoencoder using Keras.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the input image (H, W, C).
        filters (list): List of filters for encoder conv layers.
        latent_dims (tuple): Shape of the latent space (H, W, C).

    Returns:
        encoder (keras.Model): The encoder model.
        decoder (keras.Model): The decoder model.
        auto (keras.Model): The full autoencoder model.
    """
    # ENCODER
    input_encoder = keras.Input(shape=input_dims)
    x = input_encoder
    for f in filters:
        x = keras.layers.Conv2D(filters=f,
                                kernel_size=(3, 3),
                                padding='same',
                                activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                      padding='same')(x)
    encoder = keras.Model(inputs=input_encoder, outputs=x, name="encoder")

    # DECODER
    input_decoder = keras.Input(shape=latent_dims)
    x = input_decoder
    for f in filters[::-1]:
        x = keras.layers.Conv2D(filters=f,
                                kernel_size=(3, 3),
                                padding='same',
                                activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Final conv to reduce to original channel count
    x = keras.layers.Conv2D(filters=input_dims[-1],
                            kernel_size=(3, 3),
                            padding='same',
                            activation='sigmoid')(x)

    decoder = keras.Model(inputs=input_decoder, outputs=x, name="decoder")

    # AUTOENCODER
    input_auto = keras.Input(shape=input_dims)
    encoded = encoder(input_auto)
    decoded = decoder(encoded)
    auto = keras.Model(inputs=input_auto,
                       outputs=decoded,
                       name="conv_autoencoder")

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
