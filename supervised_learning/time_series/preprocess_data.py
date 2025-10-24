#!/usr/bin/env python3
"""
Preprocessing utilities for BTC forecasting.

This module loads raw minute-level BTC data from Bitstamp and Coinbase,
cleans and merges them, resamples to hourly OHLCV, and writes an hourly
dataset to CSV for model training.
"""
from __future__ import annotations
from plotly.offline import iplot
import plotly.graph_objs as go

import os
from typing import Optional

import numpy as np
import pandas as pd


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and ensure correct dtypes.

    Args:
        df (pd.DataFrame): Input dataframe expected to contain raw columns
            such as 'Timestamp', 'Open', 'High', 'Low', 'Close',
            'Volume_(BTC)', 'Volume_(Currency)', and 'Weighted_Price'.

    Returns:
        pd.DataFrame: Dataframe indexed by UTC `timestamp` with numeric
        columns standardized to: open, high, low, close, volume_btc,
        volume_currency, vwap.

    Raises:
        ValueError: If a 'timestamp' column cannot be found after renaming.
    """
    rename_map = {
        'Timestamp': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume_(BTC)': 'volume_btc',
        'Volume_(Currency)': 'volume_currency',
        'Weighted_Price': 'vwap',
    }
    # Only rename known columns present
    df = df.rename(
        columns={
            k: v for k,
            v in rename_map.items() if k in df.columns})

    if 'timestamp' not in df.columns:
        raise ValueError('Input dataframe must contain a Timestamp column')

    # Convert to datetime and set index
    df['timestamp'] = pd.to_datetime(
        df['timestamp'], unit='s', utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp']).set_index('timestamp')

    # Ensure numeric dtypes for the rest
    numeric_cols = [c for c in df.columns if c != 'timestamp']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    return df


def _resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample minute-level data to hourly OHLCV.

    Args:
        df (pd.DataFrame): Minute-level dataframe indexed by datetime with
            standardized columns.

    Returns:
        pd.DataFrame: Hourly dataframe with aggregated columns where open is
        first, high is max, low is min, close is last, volumes are summed,
        and vwap is the mean.
    """
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume_btc': 'sum',
        'volume_currency': 'sum',
        'vwap': 'mean',
    }
    # Keep only columns that exist
    present_agg = {k: v for k, v in agg.items() if k in df.columns}
    hourly = df.resample('1H').agg(present_agg)
    return hourly


def _merge_exchanges(bitstamp_h: pd.DataFrame,
                     coinbase_h: pd.DataFrame) -> pd.DataFrame:
    """Merge hourly data from two exchanges.

    Averages price fields across exchanges and sums volume fields.

    Args:
        bitstamp_h (pd.DataFrame): Bitstamp hourly dataframe.
        coinbase_h (pd.DataFrame): Coinbase hourly dataframe.

    Returns:
        pd.DataFrame: Merged hourly dataframe on the union of timestamps,
        with averaged prices (open, high, low, close, vwap) and summed
        volumes (volume_btc, volume_currency).
    """
    all_index = bitstamp_h.index.union(coinbase_h.index).sort_values()
    bs = bitstamp_h.reindex(all_index)
    cb = coinbase_h.reindex(all_index)

    def _mean_series(a: Optional[pd.Series],
                     b: Optional[pd.Series]) -> pd.Series:
        if a is None and b is None:
            return pd.Series(index=all_index, dtype='float64')
        if a is None:
            return b
        if b is None:
            return a
        return pd.concat([a, b], axis=1).mean(axis=1, skipna=True)

    out = pd.DataFrame(index=all_index)
    # Prices
    for col in ['open', 'high', 'low', 'close', 'vwap']:
        a = bs[col] if col in bs.columns else None
        b = cb[col] if col in cb.columns else None
        out[col] = _mean_series(a, b)

    # Volumes: sum across exchanges
    for col in ['volume_btc', 'volume_currency']:
        a = bs[col] if col in bs.columns else None
        b = cb[col] if col in cb.columns else None
        if a is None and b is None:
            out[col] = np.nan
        elif a is None:
            out[col] = b
        elif b is None:
            out[col] = a
        else:
            out[col] = a.fillna(0) + b.fillna(0)

    # Drop hours without a close price
    out = out.dropna(subset=['close'])
    return out


def preprocess_data(bitstamp_csv_path: str,
                    coinbase_csv_path: str,
                    output_csv_path: Optional[str] = None) -> pd.DataFrame:
    """Create an hourly BTC dataset from raw minute-level CSVs.

    Args:
        bitstamp_csv_path (str): Path to Bitstamp minute-level CSV.
        coinbase_csv_path (str): Path to Coinbase minute-level CSV.
        output_csv_path (Optional[str]): If provided, write the merged hourly
            dataframe to this CSV.

    Returns:
        pd.DataFrame: Hourly dataframe with columns open, high, low, close,
        volume_btc, volume_currency, vwap. The returned dataframe is indexed by
        hourly UTC timestamps; if saved, the index is reset and a 'timestamp'
        column is included in the CSV.
    """
    bs_raw = pd.read_csv(bitstamp_csv_path)
    cb_raw = pd.read_csv(coinbase_csv_path)

    bs = _standardize_columns(bs_raw)
    cb = _standardize_columns(cb_raw)

    bs_h = _resample_hourly(bs)
    cb_h = _resample_hourly(cb)

    hourly = _merge_exchanges(bs_h, cb_h)

    # Reset index for CSV friendliness
    hourly_out = hourly.copy()
    hourly_out = hourly_out.reset_index()
    hourly_out.rename(columns={'index': 'timestamp'}, inplace=True)

    if output_csv_path is not None:
        hourly_out.to_csv(output_csv_path, index=False)

    return hourly_out


if __name__ == '__main__':
    # Default to files colocated with this script
    here = os.path.dirname(os.path.abspath(__file__))
    bitstamp_path = os.path.join(
        here,
        'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
    coinbase_path = os.path.join(
        here,
        'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')

    output_path = os.path.join(here, 'preprocess_data.csv')
    preprocess_data(bitstamp_path, coinbase_path, output_path)


"""Helper plotting and legacy preprocessing utilities for BTC datasets."""


def plot_time_series(data):
    """Plot OHLC time series lines.

    Args:
        data (pd.DataFrame): Dataframe indexed by datetime containing columns
            'Open', 'High', 'Low', and 'Close'.
    """
    # create trace for each column
    trace1 = go.Scatter(
        x=data.index,
        y=data['Open'].astype(float),
        mode='lines',
        name='Open'
    )
    trace2 = go.Scatter(
        x=data.index,
        y=data['High'].astype(float),
        mode='lines',
        name='High'
    )
    trace3 = go.Scatter(
        x=data.index,
        y=data['Low'].astype(float),
        mode='lines',
        name='Low'
    )
    trace4 = go.Scatter(
        x=data.index,
        y=data['Close'].astype(float),
        mode='lines',
        name='Close'
    )

    # layout
    layout = dict(
        title='Historical Bitcoin Price',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=12, label='1y', step='month', stepmode='backward'),
                    dict(count=36, label='3y', step='month', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(visible=True),
            type='date'
        ))

    # create graph
    dataplot = [trace1, trace2, trace3, trace4]
    fig = dict(data=dataplot, layout=layout)

    iplot(fig)


def preprocess_data(path_file1, path_file2):
    """Legacy preprocessing pipeline combining two CSVs.

    Loads two CSVs, aligns on timestamp, drops unused columns, fills
    missing values from the second dataset, forward-fills remaining OHLC
    gaps, saves a CSV, and plots the resulting series.

    Args:
        path_file1 (str): Path to the first dataset (e.g., Bitstamp CSV).
        path_file2 (str): Path to the second dataset (e.g., Coinbase CSV).

    Returns:
        pd.DataFrame: Preprocessed dataframe after merging and filling values.
    """
    # existing file ?
    if not os.path.isfile(path_file1):
        raise FileNotFoundError(f"File {path_file1} doesn't exist.")
    if not os.path.isfile(path_file2):
        raise FileNotFoundError(f"File {path_file2} doesn't exist.")

    # load data
    print(f"Load data from {path_file1} and {path_file2}")
    df1 = pd.read_csv(path_file1)
    df2 = pd.read_csv(path_file2)

    # convert Timestamp
    df1 = df1.set_index(pd.to_datetime(df1['Timestamp'], unit='s'))
    df1 = df1.drop('Timestamp', axis=1)
    df2 = df2.set_index(pd.to_datetime(df2['Timestamp'], unit='s'))
    df2 = df2.drop('Timestamp', axis=1)

    # remove unused column
    del_col = ['Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']
    df1_clean = df1.drop(columns=del_col)
    df2_clean = df2.drop(columns=del_col)

    # use value in dataset2 to implement missing value of bitstamp
    combined_df = df1_clean.combine_first(df2_clean)

    # filter data after 2017
    combined_df2017 = combined_df[combined_df.index >=
                                  pd.Timestamp(2017, 1, 1)]

    # fix missing value for Open, high, low close column : continuous
    # timeseries
    combined_df2017['Open'] = combined_df2017['Open'].fillna(method='ffill')
    combined_df2017['High'] = combined_df2017['High'].fillna(method='ffill')
    combined_df2017['Low'] = combined_df2017['Low'].fillna(method='ffill')
    combined_df2017['Close'] = combined_df2017['Close'].fillna(method='ffill')

    # save dataset
    combined_df2017.to_csv('preprocess_data.csv', index=False)

    plot_time_series(combined_df2017)

    return combined_df2017


if __name__ == "__main__":
    preprocessed_data = preprocess_data("bitstamp.csv", "coinbase.csv")
