#!/usr/bin/env python3
"""Timestamp renaming and conversion utilities."""

import pandas as pd


def rename(df):
    """Rename a timestamp column and keep price data.

    The ``Timestamp`` column is renamed to ``Datetime`` and converted from
    seconds since the Unix epoch to pandas datetime. Only ``Datetime`` and
    ``Close`` columns are kept.

    Args:
        df (pandas.DataFrame): Input time series data.

    Returns:
        pandas.DataFrame: Modified DataFrame with two columns.
    """
    df.rename(columns={"Timestamp": "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    return df[["Datetime", "Close"]]
