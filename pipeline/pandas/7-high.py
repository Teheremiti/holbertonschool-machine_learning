#!/usr/bin/env python3
"""Sorting helpers for price data."""


def high(df):
    """Sort rows by the ``High`` price in descending order.

    Args:
        df (pandas.DataFrame): Input price data.

    Returns:
        pandas.DataFrame: Sorted DataFrame.
    """
    return df.sort_values(by="High", ascending=False)
