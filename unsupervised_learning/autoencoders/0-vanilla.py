#!/usr/bin/env python3
"""
This module defines an autoencoder model using Keras.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder.

    Args:
        input_dims (int): Dimensionality of the input.
        hidden_layers (list): List of nodes for each hidden layer in
        the encoder.
        latent_dims (int): Dimensionality of the latent space representation.

    Returns:
        encoder (keras.Model): Encoder model.
        decoder (keras.Model): Decoder model.
        auto (keras.Model): Full autoencoder model.
    """
    # ENCODER
    input_encoder = keras.Input(shape=(input_dims,))
    x = input_encoder
    for nodes in hidden_layers:
        x = keras.layers.Dense(units=nodes, activation='relu')(x)
    latent = keras.layers.Dense(units=latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs=input_encoder, outputs=latent, name="encoder")

    # DECODER
    input_decoder = keras.Input(shape=(latent_dims,))
    x = input_decoder
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(units=nodes, activation='relu')(x)
    output = keras.layers.Dense(units=input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=input_decoder, outputs=output, name="decoder")

    # AUTOENCODER
    input_auto = keras.Input(shape=(input_dims,))
    encoded = encoder(input_auto)
    decoded = decoder(encoded)
    auto = keras.Model(inputs=input_auto, outputs=decoded, name="autoencoder")

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
