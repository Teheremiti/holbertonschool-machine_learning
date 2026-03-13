#!/usr/bin/env python3
"""Concatenate Bitstamp and Coinbase data with a hierarchical index."""

import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """Concatenate two exchanges with a MultiIndex.

    All rows from ``df2`` (Bitstamp) with ``Timestamp`` up to 1417411920 are
    kept and placed before all rows from ``df1`` (Coinbase). Both frames are
    indexed by ``Timestamp`` and concatenated with a source key level.

    Args:
        df1 (pandas.DataFrame): Coinbase data.
        df2 (pandas.DataFrame): Bitstamp data.

    Returns:
        pandas.DataFrame: Concatenated DataFrame with a two-level index.
    """
    df2 = df2.loc[df2["Timestamp"] <= 1417411920]
    df1 = index(df1)
    df2 = index(df2)
    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
