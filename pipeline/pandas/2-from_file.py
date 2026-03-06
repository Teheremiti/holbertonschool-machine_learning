#!/usr/bin/env python3
"""From File"""

import pandas as pd


def from_file(filename, delimiter):
    """Load data from a delimited text file into a pandas DataFrame.

    Args:
        filename (str): Path to the input file.
        delimiter (str): Column separator used in the file.

    Returns:
        pandas.DataFrame: DataFrame containing the loaded data.
    """
    return pd.read_csv(filename, delimiter=delimiter)
