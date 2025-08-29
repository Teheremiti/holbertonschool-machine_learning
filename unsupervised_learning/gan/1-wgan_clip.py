#!/usr/bin/env python3
"""
Wasserstein GAN with weight clipping (WGAN-Clip).
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGANClip(keras.Model):
    """
    Wasserstein Generative Adversarial Network (WGAN) with weight clipping.

    This model implements the WGAN training procedure, where the
    discriminator (critic) weights are clipped between -1 and 1.
    """

    def __init__(self,
                 generator,
                 discriminator,
                 latent_generator,
                 real_examples,
                 batch_size: int = 200,
                 disc_iter: int = 2,
                 learning_rate: float = 0.005):
        """
        Initialize the WGAN-Clip model.

        Args:
            generator (keras.Model): The generator network.
            discriminator (keras.Model): The discriminator (critic) network.
            latent_generator (Callable): Function that generates latent
            vectors.
            real_examples (tf.Tensor): Real dataset samples for training.
            batch_size (int, optional): Number of samples per batch.
            disc_iter (int, optional): Number of discriminator updates
            per step.
            learning_rate (float, optional): Optimizer learning rate.
        """
        super().__init__()

        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate

        # Adam optimizer hyperparameters
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # Define generator loss and optimizer
        self.generator.loss = lambda fake_output: -tf.reduce_mean(fake_output)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        # Define discriminator loss and optimizer
        self.discriminator.loss = (
            lambda real_output, fake_output:
            tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss
        )

    def get_fake_sample(self, size: int = None, training: bool = False):
        """
        Generate fake samples using the generator.

        Args:
            size (int, optional): Number of fake samples.
            training (bool, optional): Whether to run the generator
            in training mode.

        Returns:
            tf.Tensor: Batch of fake samples.
        """
        if size is None:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size: int = None):
        """
        Sample real data from the dataset.

        Args:
            size (int, optional): Number of real samples to fetch.

        Returns:
            tf.Tensor: Batch of real samples.
        """
        if size is None:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, data):
        """
        Custom training step for WGAN with weight clipping.

        Args:
            data (Any): Not used. Training is based on real_examples stored
            in the model.

        Returns:
            dict: Dictionary containing discriminator and generator losses.
        """
        # Train discriminator (critic) multiple times
        for _ in range(self.disc_iter):
            real_sample = self.get_real_sample()
            fake_sample = self.get_fake_sample()

            with tf.GradientTape() as disc_tape:
                disc_real_output = self.discriminator(real_sample)
                disc_fake_output = self.discriminator(fake_sample)

                discr_loss = self.discriminator.loss(
                    real_output=disc_real_output,
                    fake_output=disc_fake_output
                )

            disc_gradients = disc_tape.gradient(
                discr_loss,
                self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(disc_gradients, self.discriminator.trainable_variables)
            )

            # Clip discriminator weights between [-1, 1]
            for weight in self.discriminator.trainable_weights:
                weight.assign(tf.clip_by_value(weight, -1.0, 1.0))

        # Train generator once
        with tf.GradientTape() as gen_tape:
            fake_sample = self.get_fake_sample(training=True)
            gen_out = self.discriminator(fake_sample, training=False)
            gen_loss = self.generator.loss(gen_out)

        gen_gradients = gen_tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
