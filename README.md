# GDP Forecasting for Albania using time series and machine learning.

This repository contains code for GDP forecasting of Albania using various time series models. The project utilizes Python and libraries such as Pandas, NumPy, Matplotlib, statsmodels, and pandas_datareader.

## Dependencies

- pandas
- numpy
- matplotlib
- statsmodels
- pandas_datareader

You can install these packages using `pip`:

## Overview

The code in this repository performs the following tasks:

1. Fetches real-time GDP data for Albania from 1980-2022 using the World Bank API.
2. Interpolates missing values and decomposes the time series into its components: trend, seasonality, and residual.
3. Forecasts GDP using different time series models: ARIMA, Simple Exponential Smoothing (SES), and Seasonal Autoregressive Integrated Moving Average (SARIMA).
4. Compares the forecast errors (MAE, MSE, and RMSE) of each model to determine the best one for GDP forecasting.

## Usage

To run the code, simply execute the Python script:


This will generate plots of the original data, the decomposed time series, and the forecasts for each model. The best model for GDP forecasting, along with its corresponding MAE, MSE, and RMSE values, will be printed to the console.

## Results

The generated plots will be saved as image files in the current directory:

- gdp_al_1980-2021.png: Decomposed time series components
- gdp_al_1980-2026_ses.png: GDP forecast using SES
- gdp_al_1980-2026_arima.png: GDP forecast using ARIMA
- gdp_al_1980-2026_sarima.png: GDP forecast using SARIMA

The best model for GDP forecasting will be determined based on the lowest MAE, MSE, and RMSE values.