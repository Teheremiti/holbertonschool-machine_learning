#!/usr/bin/env python3
"""Clean, aggregate, and plot Bitcoin price and volume over time."""

import matplotlib.pyplot as plt
import pandas as pd

from_file = __import__('2-from_file').from_file

df = from_file("coinbase.csv", ",")

df.drop(columns=["Weighted_Price"], inplace=True)
df.rename(columns={"Timestamp": "Date"}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"], unit="s")
df["Date"] = df["Date"].dt.to_period("d")
df = df.loc[df["Date"] >= "2017-01-01"]

df = df.set_index("Date")

df["Close"] = df["Close"].ffill()
df[["High", "Low", "Open"]] = df[["High", "Low", "Open"]].fillna(
    df["Close"]
)
df[["Volume_(BTC)", "Volume_(Currency)"]] = df[
    ["Volume_(BTC)", "Volume_(Currency)"]
].fillna(0)

df = df.resample("D").agg(
    {
        "Open": "mean",
        "High": "max",
        "Low": "min",
        "Close": "mean",
        "Volume_(BTC)": "sum",
        "Volume_(Currency)": "sum",
    }
)

df = df[df.index.year >= 2017]

plt.figure(figsize=(12, 6))
df.plot(x_compat=True)
plt.show()
