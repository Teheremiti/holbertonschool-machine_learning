#!/usr/bin/env python3
"""Helpers for working with hierarchical time series indices."""

import pandas as pd

index = __import__("10-index").index


def hierarchy(df1, df2):
    """Build a hierarchical view with timestamp as the first level.

    Data from both exchanges between timestamps 1417411980 and 1417417980
    (inclusive) is selected, indexed by ``Timestamp`` and concatenated with
    a source key. The resulting MultiIndex is reordered so that timestamp is
    the first level and the data is sorted chronologically.

    Args:
        df1 (pandas.DataFrame): Coinbase data.
        df2 (pandas.DataFrame): Bitstamp data.

    Returns:
        pandas.DataFrame: Hierarchically indexed and sorted data.
    """
    df1 = df1.loc[
        (df1["Timestamp"] >= 1417411980)
        & (df1["Timestamp"] <= 1417417980)
    ]
    df2 = df2.loc[
        (df2["Timestamp"] >= 1417411980)
        & (df2["Timestamp"] <= 1417417980)
    ]

    df1 = index(df1)
    df2 = index(df2)

    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
    df = df.reorder_levels([1, 0], axis=0)
    return df.sort_index()
