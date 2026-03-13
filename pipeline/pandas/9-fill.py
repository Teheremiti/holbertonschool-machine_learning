#!/usr/bin/env python3
"""Missing-value handling helpers for price data."""


def fill(df):
    """Remove ``Weighted_Price`` and fill missing values.

    Missing ``Close`` values are forward-filled. Missing ``High``, ``Low``
    and ``Open`` values are replaced by the row's ``Close`` value. Missing
    ``Volume_(BTC)`` and ``Volume_(Currency)`` values are set to zero.

    Args:
        df (pandas.DataFrame): Input price data.

    Returns:
        pandas.DataFrame: Modified DataFrame with no missing values in the
        handled columns.
    """
    df.drop(columns=["Weighted_Price"], inplace=True)

    # Forward-fill missing Close values
    df["Close"] = df["Close"].ffill()

    # For each price column, fill missing values with the row's Close value
    for col in ["High", "Low", "Open"]:
        df[col] = df[col].fillna(df["Close"])

    # Volumes: replace missing values with zero
    df[["Volume_(BTC)", "Volume_(Currency)"]] = df[
        ["Volume_(BTC)", "Volume_(Currency)"]
    ].fillna(0)

    return df
