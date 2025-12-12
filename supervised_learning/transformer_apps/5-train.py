#!/usr/bin/env python3
"""Transformer training module for Portuguese to English translation."""
import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate schedule from the original Transformer paper.

    Implements: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """

    def __init__(self, dm, warmup_steps=4000):
        """Initializes the CustomSchedule.

        Args:
            dm: int, dimensionality of the model.
            warmup_steps: int, number of warmup steps.
        """
        super().__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Computes the learning rate for a given training step.

        Args:
            step: The current training step number.

        Returns:
            tf.float32: The learning rate value.
        """
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Creates and trains a transformer for Portuguese to English translation.

    Args:
        N: int, number of blocks in the encoder and decoder.
        dm: int, dimensionality of the model.
        h: int, number of attention heads.
        hidden: int, number of hidden units in the fully connected layers.
        max_len: int, maximum number of tokens per sequence.
        batch_size: int, batch size for training.
        epochs: int, number of epochs to train for.

    Returns:
        Transformer: The trained transformer model.
    """
    # Load dataset
    data = Dataset(batch_size, max_len)

    # Create transformer model
    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N=N,
        dm=dm,
        h=h,
        hidden=hidden,
        input_vocab=input_vocab,
        target_vocab=target_vocab,
        max_seq_input=max_len,
        max_seq_target=max_len
    )

    # Set up optimizer with custom learning rate schedule
    learning_rate = CustomSchedule(dm, warmup_steps=4000)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    # Loss function with masking for padding tokens
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )

    def loss_function(real, pred):
        """Computes masked sparse categorical cross-entropy loss."""
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    # Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    def accuracy_function(real, pred):
        """Computes masked accuracy."""
        accuracies = tf.equal(
            real, tf.argmax(
                pred, axis=2, output_type=tf.int64))
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)
        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    @tf.function
    def train_step(inp, tar):
        """Executes a single training step."""
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            inp, tar_inp
        )

        with tf.GradientTape() as tape:
            predictions, _ = transformer(
                inp, tar_inp, True,
                enc_padding_mask, combined_mask, dec_padding_mask
            )
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables)
        )

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    # Training loop
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch, (inp, tar) in enumerate(data.data_train):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(
                    f'Epoch {epoch + 1}, batch {batch}: '
                    f'loss {train_loss.result():.4f} '
                    f'accuracy {train_accuracy.result():.4f}'
                )

        print(
            f'Epoch {epoch + 1}: '
            f'loss {train_loss.result():.4f} '
            f'accuracy {train_accuracy.result():.4f}'
        )

    return transformer
