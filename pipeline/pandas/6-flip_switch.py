#!/usr/bin/env python3
"""Transpose and column-sorting helpers for price data."""


def flip_switch(df):
    """Transpose data and sort columns in reverse order.

    The DataFrame is transposed and its columns are sorted in descending
    order by column label.

    Args:
        df (pandas.DataFrame): Input price data.

    Returns:
        pandas.DataFrame: Transposed and sorted DataFrame.
    """
    transposed = df.T
    return transposed.sort_index(axis=1, ascending=False)
