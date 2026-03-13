#!/usr/bin/env python3
"""Index helpers for time series data."""


def index(df):
    """Set the ``Timestamp`` column as the index.

    Args:
        df (pandas.DataFrame): Input time series data.

    Returns:
        pandas.DataFrame: DataFrame indexed by ``Timestamp``.
    """
    return df.set_index("Timestamp")
