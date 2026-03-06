#!/usr/bin/env python3
"""Row-slicing helpers for price data."""


def slice(df):
    """Extract selected columns every 60th row.

    The columns ``High``, ``Low``, ``Close`` and ``Volume_(BTC)`` are kept,
    and every 60th row starting from the first is selected.

    Args:
        df (pandas.DataFrame): Input price data.

    Returns:
        pandas.DataFrame: Sliced DataFrame.
    """
    columns = ["High", "Low", "Close", "Volume_(BTC)"]
    return df[columns].iloc[::60]
