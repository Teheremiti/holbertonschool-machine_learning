#!/usr/bin/env python3
"""Row-filtering helpers for price data."""


def prune(df):
    """Remove rows where ``Close`` is missing.

    Args:
        df (pandas.DataFrame): Input price data.

    Returns:
        pandas.DataFrame: DataFrame without ``NaN`` in ``Close``.
    """
    return df.dropna(subset=["Close"])
