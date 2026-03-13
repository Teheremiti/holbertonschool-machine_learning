#!/usr/bin/env python3
"""Descriptive statistics helpers for price data."""


def analyze(df):
    """Compute descriptive statistics for numeric columns.

    The ``Timestamp`` column, if present, is set as the index before
    computing statistics so it is not included in the result.

    Args:
        df (pandas.DataFrame): Input data.

    Returns:
        pandas.DataFrame: Summary statistics for numeric columns.
    """
    if "Timestamp" in df.columns:
        df = df.set_index("Timestamp")
    return df.describe(include="number")
