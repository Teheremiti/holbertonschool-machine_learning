#!/usr/bin/env python3
"""
This module defines a modular variational autoencoder using Keras.
"""

import tensorflow.keras as keras


class Sampling(keras.layers.Layer):
    """Reparameterization trick layer."""

    def call(self, inputs):
        """Samples from the latent space."""
        mean, log_var = inputs
        eps = keras.backend.random_normal(
            shape=keras.backend.shape(mean))
        return mean + keras.backend.exp(0.5 * log_var) * eps


def build_encoder(input_dims, hidden_layers, latent_dims):
    """
    Builds the encoder network.

    Args:
        input_dims (int): Input dimension.
        hidden_layers (list): Sizes of hidden layers.
        latent_dims (int): Size of latent space.

    Returns:
        keras.Model: Encoder model outputting z, mean, log_var.
    """
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    mean = keras.layers.Dense(latent_dims)(x)
    log_var = keras.layers.Dense(latent_dims)(x)
    z = Sampling()([mean, log_var])
    return keras.Model(inputs=inputs,
                       outputs=[z, mean, log_var],
                       name="encoder")


def build_decoder(hidden_layers, latent_dims, output_dims):
    """
    Builds the decoder network.

    Args:
        hidden_layers (list): Sizes of hidden layers (reversed).
        latent_dims (int): Input latent space dimension.
        output_dims (int): Output shape.

    Returns:
        keras.Model: Decoder model.
    """
    inputs = keras.Input(shape=(latent_dims,))
    x = inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(output_dims, activation='sigmoid')(x)
    return keras.Model(inputs=inputs,
                       outputs=outputs,
                       name="decoder")


class VAE(keras.Model):
    """Custom Variational Autoencoder."""

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def compile(self, optimizer, loss):
        super().compile(optimizer=optimizer, loss=loss)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with keras.backend.name_scope("vae_train"):
            with keras.backend.get_graph().gradient_tape() as tape:
                z, mean, log_var = self.encoder(data)
                reconstructed = self.decoder(z)
                rec = keras.losses.binary_crossentropy(data,
                                                       reconstructed)
                rec = keras.backend.sum(rec, axis=1)
                kl = -0.5 * keras.backend.sum(
                    1 + log_var - keras.backend.square(mean) -
                    keras.backend.exp(log_var), axis=1)
                total = keras.backend.mean(rec + kl)
            grads = tape.gradient(total, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads,
                                               self.trainable_variables))
        return {"loss": total}

    def call(self, inputs):
        z, _, _ = self.encoder(inputs)
        return self.decoder(z)


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): Input dimension.
        hidden_layers (list): Encoder hidden layers.
        latent_dims (int): Latent space dimension.

    Returns:
        encoder, decoder, auto (VAE model)
    """
    encoder = build_encoder(input_dims,
                            hidden_layers,
                            latent_dims)
    decoder = build_decoder(hidden_layers,
                            latent_dims,
                            input_dims)
    vae = VAE(encoder=encoder,
              decoder=decoder)
    vae.compile(optimizer='adam',
                loss=keras.losses.BinaryCrossentropy())
    return encoder, decoder, vae
