import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas_datareader import wb
from matplotlib.ticker import FuncFormatter

# Task 1

# Fetch real-time GDP data for Albania from 1980-2022
gdp_data = wb.download(indicator="NY.GDP.MKTP.KD", country="AL", start="1980", end="2021")
gdp_data = gdp_data.reset_index().pivot(index="year", columns="country", values="NY.GDP.MKTP.KD")
gdp_data.index = pd.to_datetime(gdp_data.index, format="%Y")

# print(gdp_data)

# Interpolate missing values
gdp_data.interpolate(method="linear", inplace=True)

# Decompose the time series
decomposition = seasonal_decompose(gdp_data, model="multiplicative")

# Function to format y-axis tick labels
def dynamic_formatter(x, pos):
    if abs(x) >= 1e9:
        return f"{x / 1e9:.1f}B"
    elif abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    elif abs(x) >= 1e3:
        return f"{x / 1e3:.1f}K"
    else:
        return f"{x:.1f}"

formatter = FuncFormatter(dynamic_formatter)

# Customize the plot
plt.style.use("seaborn-v0_8-darkgrid")
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

axes[0].plot(gdp_data, label="Original", linewidth=2)
axes[0].set_title("GDP Data for Albania")
axes[0].legend()
axes[0].yaxis.set_major_formatter(formatter)

axes[1].plot(decomposition.trend, label="Trend", linewidth=2)
axes[1].legend()
axes[1].yaxis.set_major_formatter(formatter)

axes[2].plot(decomposition.seasonal, label="Seasonal", linewidth=2)
axes[2].legend()

axes[3].plot(decomposition.resid, label="Residual", linewidth=2)
axes[3].legend()
axes[3].yaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.savefig("gdp_al_1980-2021.png", dpi=300)

# plt.show()


## Task 2

from statsmodels.tsa.arima.model import ARIMA

# Fit the ARIMA model (p, d, q)
model_arima = ARIMA(gdp_data, order=(1, 1, 1)).fit()

# Forecast the next 5 years
forecast_arima = model_arima.forecast(steps=5)

# print("Data forecasted for the next 5 years using ARIMA (Autoregressive Integrated Moving Average) \n", forecast_arima)

####

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Fit the Simple Exponential Smoothing model
model_ses = SimpleExpSmoothing(gdp_data).fit()

# Forecast the next 5 years
forecast_ses = model_ses.forecast(5)

# print("Data forecasted for the next 5 years using SES (SimpleExponentialSmoothing) \n", forecast_arima)

## Generating images for each of the individual plots:
def plot_forecast(data, forecast, method, title, ylabel, filepath):
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Original", linewidth=2)
    plt.plot(forecast, label="Forecast", linewidth=2)
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    # plt.show()

# Plot and save SES forecast
plot_forecast(gdp_data, forecast_ses, "SES", "GDP Data for Albania - SES Forecast", "GDP in USD", "gdp_al_1980-2026_ses.png")

# Plot and save ARIMA forecast
plot_forecast(gdp_data, forecast_arima, "ARIMA", "GDP Data for Albania - ARIMA Forecast", "GDP in USD", "gdp_al_1980-2026_arima.png")

###

## Task 3

'''
Adding a better forecasting method called SARIMA (Seasonal Autoregressive Integrated Moving Average) which takes seasonality into account. We will also create a function to plot ex post and ex ante
forecasts in relation to actual data.
'''

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit the SARIMA model (p, d, q) x (P, D, Q, s)
model_sarima = SARIMAX(gdp_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4)).fit()

# Forecast the next 5 years
forecast_sarima = model_sarima.forecast(steps=5)

print("Data forecasted for the next 5 years using SARIMA (Seasonal Autoregressive Integrated Moving Average) \n", forecast_sarima)


# Creating a function to plot ex post and ex ante forecasts in relation to actual data:

def plot_ex_post_and_ex_ante(data, forecast, method, title, ylabel, filepath):
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Original", linewidth=2)
    plt.plot(forecast, label=f"{method} Forecast", linewidth=2)
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    # plt.show()


# Plotting the data

plot_ex_post_and_ex_ante(gdp_data, forecast_sarima, "SARIMA", "GDP Data for Albania - SARIMA Forecast", "GDP in USD", "gdp_al_1980-2026_sarima.png")


# Forecasting

from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_forecast_errors(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

'''Splitting the data into training and test set. From 1980 to 2016 we have 
the data for training and the data from 2017-2021 for testing. 
'''

train_data = gdp_data.loc["1980":"2016"]
test_data = gdp_data.loc["2017":"2021"]

# ARIMA model
model_arima = ARIMA(train_data, order=(1, 1, 1)).fit()
forecast_arima = model_arima.forecast(steps=len(test_data))

# SES model
model_ses = SimpleExpSmoothing(train_data).fit()
forecast_ses = model_ses.forecast(len(test_data))

# SARIMA model
model_sarima = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4)).fit()
forecast_sarima = model_sarima.forecast(steps=len(test_data))


mae_arima, mse_arima, rmse_arima = compute_forecast_errors(test_data, forecast_arima)
mae_ses, mse_ses, rmse_ses = compute_forecast_errors(test_data, forecast_ses)
mae_sarima, mse_sarima, rmse_sarima = compute_forecast_errors(test_data, forecast_sarima)

# print("ARIMA forecast errors - MAE: {:.2f}, MSE: {:.2f}, RMSE: {:.2f}".format(mae_arima, mse_arima, rmse_arima))
# print("SES forecast errors - MAE: {:.2f}, MSE: {:.2f}, RMSE: {:.2f}".format(mae_ses, mse_ses, rmse_ses))
# print("SARIMA forecast errors - MAE: {:.2f}, MSE: {:.2f}, RMSE: {:.2f}".format(mae_sarima, mse_sarima, rmse_sarima))

# Compare the forecast errors to determine the best model
best_model = None
best_mae = min(mae_arima, mae_ses, mae_sarima)
best_mse = min(mse_arima, mse_ses, mse_sarima)
best_rmse = min(rmse_arima, rmse_ses, rmse_sarima)

if best_mae == mae_arima and best_mse == mse_arima and best_rmse == rmse_arima:
    best_model = "ARIMA"
elif best_mae == mae_ses and best_mse == mse_ses and best_rmse == rmse_ses:
    best_model = "SES"
elif best_mae == mae_sarima and best_mse == mse_sarima and best_rmse == rmse_sarima:
    best_model = "SARIMA"

# Print the best model and the corresponding values for MAE, MSE, and RMSE
print("The best model for GDP forecasting is: {}".format(best_model))
print("MAE: {:.2f}, MSE: {:.2f}, RMSE: {:.2f}".format(best_mae, best_mse, best_rmse))