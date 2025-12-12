#!/usr/bin/env python3
"""Full transformer model implementation."""
import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.layers.Layer):
    """Transformer model combining encoder and decoder."""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Initialize the transformer model.

        Args:
            N (int): Number of blocks in encoder and decoder.
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Hidden units in the feed-forward network.
            input_vocab (int): Size of the input vocabulary.
            target_vocab (int): Size of the target vocabulary.
            max_seq_input (int): Maximum input sequence length.
            max_seq_target (int): Maximum target sequence length.
            drop_rate (float): Dropout rate.
        """
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """Run the transformer forward pass.

        Args:
            inputs (tf.Tensor): Input tokens of shape (batch, input_seq_len).
            target (tf.Tensor): Target tokens of shape (batch, target_seq_len).
            training (bool): Whether the model is in training mode.
            encoder_mask (tf.Tensor): Mask applied to the encoder.
            look_ahead_mask (tf.Tensor): Mask applied to the decoder.
            decoder_mask (tf.Tensor): Padding mask applied to the decoder.

        Returns:
            tf.Tensor: Output logits of shape
            (batch, target_seq_len, target_vocab).
        """
        x = self.encoder(inputs, training, encoder_mask)
        x = self.decoder(target, x, training, look_ahead_mask, decoder_mask)
        output = self.linear(x)

        return output
