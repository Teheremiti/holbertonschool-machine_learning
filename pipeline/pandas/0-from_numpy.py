#!/usr/bin/env python3
"""From Numpy"""

import numpy as np
import pandas as pd


def from_numpy(array):
    """Create a DataFrame from a NumPy ndarray.

    The columns are labeled with capital letters in alphabetical order.

    Args:
        array (np.ndarray): Array used to build the DataFrame.

    Returns:
        pandas.DataFrame: DataFrame built from ``array``.

    Raises:
        TypeError: If ``array`` is not a NumPy ndarray.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Input should be an np.ndarray")

    num_cols = array.shape[1]
    columns = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:num_cols]

    return pd.DataFrame(array, columns=columns)
