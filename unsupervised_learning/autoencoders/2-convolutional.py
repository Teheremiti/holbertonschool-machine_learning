#!/usr/bin/env python3
"""Defines a convolutional autoencoder using Keras."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the model input.
        filters (list): List of filters for the encoder.
        latent_dims (tuple): Dimensions of the latent space.

    Returns:
        encoder: The encoder model.
        decoder: The decoder model.
        auto: The full autoencoder model.
    """
    # Encoder
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    encoder = keras.Model(inputs, x, name='encoder')

    # Decoder
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input
    for f in filters[:-1][::-1]:
        x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
    # Second to last conv layer: valid padding
    x = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                            padding='valid', activation='relu')(x)
    # Final output layer
    x = keras.layers.Conv2D(filters=input_dims[-1], kernel_size=(3, 3),
                            padding='same', activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, x, name='decoder')

    # Autoencoder
    auto_input = inputs
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
