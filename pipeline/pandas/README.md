# Pandas

Small collection of exercises using NumPy and pandas for time series data.

## Environment

- Python 3.9
- NumPy 1.25.2
- pandas 2.2.2

## Files

- `0-from_numpy.py`: `from_numpy` builds a DataFrame from a NumPy array.
- `1-from_dictionary.py`: builds a small DataFrame from a dictionary.
- `2-from_file.py`: `from_file` loads a delimited text file to a DataFrame.
- `3-rename.py`: renames `Timestamp` to `Datetime` and shows key columns.
- `4-array.py`: prints last ten `High` and `Close` rows as a NumPy array.
- `5-slice.py`: selects columns and keeps every 60th row.
- `6-flip_switch.py`: transposes the table and sorts columns in reverse time.
- `7-high_low.py`: sorts rows by the `High` price in descending order.
- `8-prune.py`: removes rows where `Close` is missing.
- `9-fill.py`: drops `Weighted_Price` and fills remaining missing values.
- `10-index.py`: sets `Timestamp` as the DataFrame index.
- `11-concat.py`: concatenates Bitstamp and Coinbase with a MultiIndex.
- `12-hierarchy.py`: reorders the MultiIndex so timestamp is the first
  level.
- `13-analyze.py`: computes descriptive statistics for numeric columns.
- `14-visualize.py`: cleans, aggregates, and plots daily price and volume.