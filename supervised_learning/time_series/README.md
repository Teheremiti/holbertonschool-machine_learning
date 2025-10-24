# BTC Hourly Forecasting with RNNs

## Overview
This project forecasts the next hour's Bitcoin (BTC) close price using Recurrent Neural Networks (RNNs). It:
- Preprocesses raw minute-level data from Bitstamp and Coinbase
- Resamples to hourly OHLCV
- Builds tf.data windows of the past 24 hours to predict the next hour
- Trains and evaluates LSTM and GRU models with mean squared error (MSE)

## Requirements
- Python 3.9
- TensorFlow 2.15
- NumPy 1.25.2
- Pandas 2.2.2
- Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data
Download the raw CSV files from the following links:
- [Bitstamp](https://intranet.hbtn.io/rltoken/JjaZZyvz3hChdFxPNbc3hA) (rename in bitstamp.csv)
- [Coinbase](https://intranet.hbtn.io/rltoken/vEVzC0M9D73iMNUZqf7Tpg) (rename in coinbase.csv)

Each row represents a 60-second window with: Timestamp (Unix), Open, High, Low, Close, Volume_(BTC), Volume_(Currency), and Weighted_Price.

## Preprocessing
Script: `preprocess_data.py`
- Standardizes columns to: `open, high, low, close, volume_btc, volume_currency, vwap`
- Converts Unix timestamps to UTC datetimes
- Resamples minute data to hourly OHLCV
- Merges exchanges by averaging prices and summing volumes
- Writes `preprocess_data.csv`

Run preprocessing:
```bash
cd supervised_learning/time_series
python3 preprocess_data.py
```

Output: `preprocess_data.csv` with columns `timestamp, open, high, low, close, volume_btc, volume_currency, vwap`.

## Forecasting
Script: `forecast_btc.py`
- Loads `preprocess_data.csv` (creates it if missing)
- Uses 24-hour input windows to predict the next hour's close
- Normalizes features using train-set statistics
- Builds two models: LSTM and GRU
- Trains with early stopping (MSE loss, Adam optimizer)
- Evaluates on validation and test splits (70/20/10)

Run training:
```bash
cd supervised_learning/time_series
python3 forecast_btc.py
```

Artifacts produced:
- `LSTM_model.h5` and `GRU_model.h5`
- Console metrics for validation and test sets
- Training curves (loss and MAE)

## Project Structure
- `preprocess_data.py`: Create hourly merged dataset and save CSV
- `forecast_btc.py`: Windowing, model training (LSTM/GRU), evaluation, plots
- Data CSVs: Exchange minute-level inputs (see Data section)

## Notes
- The model predicts 1 hour ahead using the past 24 hours of features
- MSE is used as the loss; MAE is reported for interpretability
- The data pipeline uses `tf.data` via `timeseries_dataset_from_array`

## Disclaimer
This project is for educational purposes only. Do not use the forecasts for financial decisions.