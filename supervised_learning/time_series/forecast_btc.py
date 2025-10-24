#!/usr/bin/env python3
"""
Train and evaluate RNN models to forecast BTC hourly close.

Uses 24 past hours to predict the next hour's close price. Data is fed via
tf.data windows and mean squared error (MSE) is used as the loss.
"""

from preprocess_data import preprocess_data as run_preprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
GRU = tf.keras.layers.GRU
EarlyStopping = tf.keras.callbacks.EarlyStopping


class WindowGenerator():
    """Generate sliding windows for time series forecasting.

    This utility slices input sequences and corresponding labels to feed an RNN
    model for next-hour BTC close prediction.

    Args:
        input_width (int): Number of past time steps used as input.
        label_width (int): Number of future time steps to predict.
        shift (int): Offset between the end of inputs and start of labels.
        train_df (pd.DataFrame): Training features.
        val_df (pd.DataFrame): Validation features.
        test_df (pd.DataFrame): Test features.
        label_columns (list[str] | None): Names of target columns.
    """

    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        """Initialize a window generator with dataset splits and sizes.

        Args:
            input_width (int): Number of past steps used as model input.
            label_width (int): Number of future steps used as labels.
            shift (int): Steps between end of inputs and start of labels.
            train_df (pd.DataFrame): Training dataframe.
            val_df (pd.DataFrame): Validation dataframe.
            test_df (pd.DataFrame): Test dataframe.
            label_columns (list[str] | None): Target column names to extract.
        """
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def __repr__(self):
        """Return a readable summary of the window configuration.

        Returns:
            str: Human-readable configuration for debugging and logging.
        """
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        """
        Split a sequence window into inputs and labels.

        Args:
            features (tf.Tensor): Batch of windows shaped
                (batch, total_window_size, num_features).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Inputs and labels tensors suitable for
            model training.
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]]
                               for name in self.label_columns], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. If predicting only next hour, drop the time axis in labels.
        inputs.set_shape([None, self.input_width, None])
        if self.label_width == 1:
            labels = labels[:, 0, :]
            labels.set_shape([None, None])
        else:
            labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        """
        Build a tf.data.Dataset of sliding windows.

        Args:
            data (pd.DataFrame | np.ndarray): Feature data.

        Returns:
            tf.data.Dataset: Dataset yielding (inputs, labels) pairs.
        """
        df_tf = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data.values if isinstance(data, pd.DataFrame) else data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=32)

        # apply time windowing to the tf.dataset
        df_tf = df_tf.map(self.split_window)

        return df_tf

    """
    The WindowGenerator object holds training, validation and test data.
    Properties addition for accessing them as tf.data.Datasets using the above make_dataset method.
    Addition of a standard example batch for easy access and plotting
    """

    @property
    def train(self):
        """tf.data.Dataset for training windows.

        Returns:
            tf.data.Dataset: Dataset yielding (inputs, labels) for training.
        """
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        """tf.data.Dataset for validation windows.

        Returns:
            tf.data.Dataset: Dataset yielding (inputs, labels) for validation.
        """
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        """tf.data.Dataset for test windows.

        Returns:
            tf.data.Dataset: Dataset yielding (inputs, labels) for testing.
        """
        return self.make_dataset(self.test_df)


def normalize_data(df_train, df_val, df_test):
    """
    Normalize datasets using statistics from the training set.

    Args:
        df_train (pd.DataFrame): Training features.
        df_val (pd.DataFrame): Validation features.
        df_test (pd.DataFrame): Test features.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Normalized train,
        validation, and test dataframes.
    """
    train_mean = df_train.mean(axis=0)
    train_std = df_train.std(axis=0)

    df_train_norm = (df_train - train_mean) / train_std
    df_val_norm = (df_val - train_mean) / train_std
    df_test_norm = (df_test - train_mean) / train_std
    return df_train_norm, df_val_norm, df_test_norm


def split_data(df):
    """
    Split a dataframe into train/validation/test sets (70/20/10).

    Args:
        df (pd.DataFrame): Feature dataframe ordered by time.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation,
        and test splits.
    """
    n = len(df)
    train_data = df.iloc[:int(n * 0.7)]
    val_data = df.iloc[int(n * 0.7): int(n * 0.9)]
    test_data = df.iloc[int(n * 0.9):]

    return train_data, val_data, test_data


def plot_eval_train(history):
    """
    Plot training and validation loss/MAE curves.

    Args:
        history (tf.keras.callbacks.History): Keras training history object.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('LSTM Model Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / MAE')
    plt.legend()
    plt.show()


def compile_fit(model, window, patience=5, epochs=200, batch_size=32):
    """
    Compile and train a model with early stopping.

    Args:
        model (tf.keras.Model): Model to train.
        window (WindowGenerator): Windowed datasets for training/validation.
        patience (int, optional): Early stopping patience. Defaults to 5.
        epochs (int, optional): Maximum number of epochs. Defaults to 200.
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        tf.keras.callbacks.History: Training history.
    """
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min',
                                   restore_best_weights=True
                                   )

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=['mae'])

    history = model.fit(window.train, epochs=epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping],
                        batch_size=batch_size,
                        verbose=1)

    print(model.summary())

    return history


if __name__ == "__main__":
    # load data and preprocess
    here = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(here, 'preprocess_data.csv')
    if not os.path.isfile(output_csv):
        bitstamp_path = os.path.join(
            here, 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
        coinbase_path = os.path.join(
            here, 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
        df_hourly = run_preprocess(bitstamp_path, coinbase_path, output_csv)
    else:
        df_hourly = pd.read_csv(output_csv)

    # split data and labels
    if 'timestamp' in df_hourly.columns:
        df_hourly = df_hourly.drop(columns=['timestamp'])

    expected_cols = ['open', 'high', 'low', 'close', 'volume_btc',
                     'volume_currency', 'vwap']
    cols_present = [c for c in expected_cols if c in df_hourly.columns]
    df_hourly = df_hourly[cols_present]

    train_df, val_df, test_df = split_data(df_hourly)

    # normalization
    train_norm, val_norm, test_norm = normalize_data(train_df, val_df, test_df)

    window = WindowGenerator(
        input_width=24, label_width=1, shift=1,
        train_df=train_norm, val_df=val_norm, test_df=test_norm,
        label_columns=['close'])

    # model
    LSTM_model = Sequential([
        LSTM(16, return_sequences=True),
        Dropout(0.5),
        LSTM(16),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    GRU_model = Sequential([
        GRU(32, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])

    history1 = compile_fit(LSTM_model, window)
    LSTM_model.save('LSTM_model.h5')
    history2 = compile_fit(GRU_model, window)
    GRU_model.save('GRU_model.h5')

    # Performance log
    val_performance = {}
    performance = {}

    # Evaluate model on validation dataset
    val_performance['LSTM'] = LSTM_model.evaluate(window.val, verbose=0)
    val_performance['GRU'] = GRU_model.evaluate(window.val, verbose=0)

    # Evaluate model on test dataset
    performance['LSTM'] = LSTM_model.evaluate(window.test, verbose=0)
    performance['GRU'] = GRU_model.evaluate(window.test, verbose=0)

    # Print results
    print("Results on validation set :")
    print(f"LSTM : {val_performance['LSTM']}")
    print(f"GRU : {val_performance['GRU']}")

    print("\nResults on test set :")
    print(f"LSTM : {performance['LSTM']}")
    print(f"GRU : {performance['GRU']}")

    print("Plot for LSTM model :")
    plot_eval_train(history1)
    print("Plot for GRU model :")
    plot_eval_train(history2)
