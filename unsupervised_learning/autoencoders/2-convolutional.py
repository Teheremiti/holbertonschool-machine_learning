#!/usr/bin/env python3
"""Convolutional Autoencoder Module"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the model input (e.g., (28, 28, 1))
        filters (list): Number of filters for each conv layer in encoder
        latent_dims (tuple): Dimensions of the latent space representation

    Returns:
        encoder (Model): Encoder model
        decoder (Model): Decoder model
        auto (Model): Full autoencoder model
    """
    # Encoder
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    encoder = keras.Model(inputs, x, name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs
    reversed_filters = filters[::-1]

    for i, f in enumerate(reversed_filters):
        if i < len(reversed_filters) - 2:
            x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                    activation='relu', padding='same')(x)
            x = keras.layers.UpSampling2D(size=(2, 2))(x)
        elif i == len(reversed_filters) - 2:
            x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                    activation='relu', padding='valid')(x)
            x = keras.layers.UpSampling2D(size=(2, 2))(x)
        else:
            x = keras.layers.Conv2D(filters=input_dims[2], kernel_size=(3, 3),
                                    activation='sigmoid', padding='same')(x)

    decoder = keras.Model(latent_inputs, x, name='decoder')

    # Autoencoder
    auto_input = keras.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
