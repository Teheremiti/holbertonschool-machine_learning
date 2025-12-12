#!/usr/bin/env python3
"""Transformer architecture implementation for sequence-to-sequence tasks."""
import numpy as np
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculates scaled dot-product attention.

    Args:
        Q: tf.Tensor of shape (..., seq_len_q, dk) containing the query matrix.
        K: tf.Tensor of shape (..., seq_len_v, dk) containing the key matrix.
        V: tf.Tensor of shape (..., seq_len_v, dv) containing the value matrix.
        mask: Optional mask where positions to mask are set to 1.

    Returns:
        tuple: (output, weights)
            - output: tf.Tensor of shape (..., seq_len_q, dv).
            - weights: tf.Tensor of shape (..., seq_len_q, seq_len_v).
    """
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(dk)

    if mask is not None:
        scaled += mask * -1e9

    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights


def positional_encoding(max_seq_len, dm):
    """Calculates positional encoding for a transformer.

    Args:
        max_seq_len: int, the maximum sequence length.
        dm: int, the model depth (embedding dimension).

    Returns:
        np.ndarray: Array of shape (max_seq_len, dm) containing the
            positional encoding vectors.
    """
    PE = np.zeros((max_seq_len, dm))
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))

    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention layer for transformer models."""

    def __init__(self, dm, h):
        """Initializes the MultiHeadAttention layer.

        Args:
            dm: int, dimensionality of the model.
            h: int, number of attention heads.
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, x, batch_size):
        """Splits the last dimension into (num_heads, depth).

        Args:
            x: tf.Tensor of shape (batch_size, seq_len, dm) to split.
            batch_size: int, the batch size.

        Returns:
            tf.Tensor: Shape (batch_size, num_heads, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Performs multi-head attention.

        Args:
            Q: tf.Tensor for queries.
            K: tf.Tensor for keys.
            V: tf.Tensor for values.
            mask: tf.Tensor, attention mask or None.

        Returns:
            tuple: (output, weights)
        """
        batch_size = tf.shape(Q)[0]

        Q = self.split_heads(self.Wq(Q), batch_size)
        K = self.split_heads(self.Wk(K), batch_size)
        V = self.split_heads(self.Wv(V), batch_size)

        scaled_att, weights = sdp_attention(Q, K, V, mask)

        scaled_att = tf.transpose(scaled_att, perm=[0, 2, 1, 3])
        concat_att = tf.reshape(scaled_att, (batch_size, -1, self.dm))

        return self.linear(concat_att), weights


class EncoderBlock(tf.keras.layers.Layer):
    """Encoder block for a transformer."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initializes the EncoderBlock.

        Args:
            dm: int, dimensionality of the model.
            h: int, number of attention heads.
            hidden: int, number of hidden units in feed-forward network.
            drop_rate: float, dropout rate.
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Processes input through the encoder block.

        Args:
            x: tf.Tensor of shape (batch, input_seq_len, dm).
            training: bool, whether in training mode.
            mask: tf.Tensor, mask for self-attention.

        Returns:
            tf.Tensor: Output of shape (batch, input_seq_len, dm).
        """
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.dense_output(self.dense_hidden(out1))
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)


class DecoderBlock(tf.keras.layers.Layer):
    """Decoder block for a transformer."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initializes the DecoderBlock.

        Args:
            dm: int, dimensionality of the model.
            h: int, number of attention heads.
            hidden: int, number of hidden units in feed-forward network.
            drop_rate: float, dropout rate.
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Processes input through the decoder block.

        Args:
            x: tf.Tensor of shape (batch, target_seq_len, dm).
            encoder_output: tf.Tensor of shape (batch, input_seq_len, dm).
            training: bool, whether in training mode.
            look_ahead_mask: tf.Tensor, mask for first attention block.
            padding_mask: tf.Tensor, mask for second attention block.

        Returns:
            tuple: (output, attn_weights1, attn_weights2)
        """
        attn1, weights1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, weights2 = self.mha2(out1, encoder_output, encoder_output,
                                    padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.dense_output(self.dense_hidden(out2))
        ffn_output = self.dropout3(ffn_output, training=training)

        return self.layernorm3(out2 + ffn_output), weights1, weights2


class Encoder(tf.keras.layers.Layer):
    """Encoder for a transformer model."""

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """Initializes the Encoder.

        Args:
            N: int, number of encoder blocks.
            dm: int, dimensionality of the model.
            h: int, number of attention heads.
            hidden: int, number of hidden units in feed-forward networks.
            input_vocab: int, size of the input vocabulary.
            max_seq_len: int, maximum sequence length.
            drop_rate: float, dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Processes input through the encoder.

        Args:
            x: tf.Tensor of shape (batch, input_seq_len).
            training: bool, whether in training mode.
            mask: tf.Tensor, mask for self-attention.

        Returns:
            tf.Tensor: Encoder output of shape (batch, input_seq_len, dm).
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """Decoder for a transformer model."""

    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """Initializes the Decoder.

        Args:
            N: int, number of decoder blocks.
            dm: int, dimensionality of the model.
            h: int, number of attention heads.
            hidden: int, number of hidden units in feed-forward networks.
            target_vocab: int, size of the target vocabulary.
            max_seq_len: int, maximum sequence length.
            drop_rate: float, dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Processes input through the decoder.

        Args:
            x: tf.Tensor of shape (batch, target_seq_len).
            encoder_output: tf.Tensor of shape (batch, input_seq_len, dm).
            training: bool, whether in training mode.
            look_ahead_mask: tf.Tensor, mask for self-attention.
            padding_mask: tf.Tensor, mask for encoder-decoder attention.

        Returns:
            tuple: (output, attention_weights)
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)

        attention_weights = {}
        for i, block in enumerate(self.blocks):
            x, w1, w2 = block(x, encoder_output, training,
                              look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i + 1}_block1'] = w1
            attention_weights[f'decoder_layer{i + 1}_block2'] = w2

        return x, attention_weights


class Transformer(tf.keras.Model):
    """Complete Transformer model for sequence-to-sequence tasks."""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Initializes the Transformer.

        Args:
            N: int, number of encoder and decoder blocks.
            dm: int, dimensionality of the model.
            h: int, number of attention heads.
            hidden: int, number of hidden units in feed-forward networks.
            input_vocab: int, size of the input vocabulary.
            target_vocab: int, size of the target vocabulary.
            max_seq_input: int, maximum input sequence length.
            max_seq_target: int, maximum target sequence length.
            drop_rate: float, dropout rate.
        """
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """Performs forward pass through the transformer.

        Args:
            inputs: tf.Tensor of shape (batch, input_seq_len).
            target: tf.Tensor of shape (batch, target_seq_len).
            training: bool, whether in training mode.
            encoder_mask: tf.Tensor, mask for encoder self-attention.
            look_ahead_mask: tf.Tensor, mask for decoder self-attention.
            decoder_mask: tf.Tensor, mask for encoder-decoder attention.

        Returns:
            tuple: (output, attention_weights)
                - output: tf.Tensor of shape (batch, target_seq_len, target_vocab)
                - attention_weights: dict of attention weights.
        """
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output, attn_weights = self.decoder(
            target, enc_output, training, look_ahead_mask, decoder_mask
        )

        return self.linear(dec_output), attn_weights
