#!/usr/bin/env python3
"""Array utilities for price data."""


def array(df):
    """Return the last ten ``High`` and ``Close`` rows as a NumPy array.

    Args:
        df (pandas.DataFrame): Input price data.

    Returns:
        numpy.ndarray: Extracted values for the last ten rows.
    """
    subset = df[["High", "Close"]].tail(10)
    return subset.to_numpy()

