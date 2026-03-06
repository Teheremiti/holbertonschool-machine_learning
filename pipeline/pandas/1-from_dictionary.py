#!/usr/bin/env python3
"""From Dictionary"""

import pandas as pd


def from_dictionary():
    """Build a simple example DataFrame.

    The resulting DataFrame has two columns and four labeled rows.

    Returns:
        pandas.DataFrame: DataFrame with columns ``First`` and ``Second``
        and index labels ``A`` through ``D``.
    """
    data = {
        "First": [0.0, 0.5, 1.0, 1.5],
        "Second": ["one", "two", "three", "four"],
    }
    df = pd.DataFrame(data, index=list("ABCD"))
    return df


df = from_dictionary()
