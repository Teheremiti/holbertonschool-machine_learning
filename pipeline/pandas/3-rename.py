#!/usr/bin/env python3
"""Rename and convert the timestamp column, then display selected data."""

import pandas as pd

from_file = __import__("2-from_file").from_file

df = from_file("coinbase.csv", ",")

df.rename(columns={"Timestamp": "Datetime"}, inplace=True)
df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
df = df[["Datetime", "Close"]]

print(df.tail())
