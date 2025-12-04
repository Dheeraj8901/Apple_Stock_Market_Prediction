
# Apple Stock Price Prediction (2012–2019)

Time-Series Forecasting | ARIMA | SARIMA | XGBoost | Streamlit Deployment

 *Project Overview

This project aims to predict Apple’s stock price for the next 30+ business days using historical stock price data (2012–2019).
It includes:

🔹 Full EDA

🔹 Statistical & ML model building

🔹 Model comparison

🔹 Residual diagnostics

🔹 Final SARIMA forecasting

🔹 Streamlit deployment with confidence intervals & model summary

🔹 Exportable predictions and visualizations

* 1. Dataset Information

Source: Apple OHLCV Data (2012–2019)
Attributes:

Column	Description
Date	Trading day
Open	Opening price
High	Daily high
Low	Daily low
Close	Final price
Adj Close	Adjusted close
Volume	Shares traded

* 2. Data Preprocessing & Feature Engineering

Converted “Date” into proper DatetimeIndex
Sorted data & removed invalid dates
Computed:

Daily returns

Log returns

21-day rolling mean

21-day rolling volatility

 Outlier handling using IQR capping
 Business-day reindexing with forward-fill for missing trading days

* 3. Exploratory Data Analysis (EDA)
  
🔹 Trend Analysis

Strong long-term upward trend in Apple prices

Non-stationarity confirmed by ADF test

🔹 Volatility

Periods of high and low volatility observed

🔹 Seasonality

Slight day-of-week effects

Mild monthly patterns

🔹 Correlations

OHLC prices almost perfectly correlated

Volume negatively correlated with price

* 4. Model Development

Three forecasting models were evaluated:

1) ARIMA (1,1,1)

Baseline time-series model

Moderate performance

2) SARIMA (1,1,1)(1,1,1,5)

Weekly seasonality (5 trading days)

Best performance across metrics

➡ Metrics:

RMSE: 8.92

MAE: 6.85

MAPE: ~2.46%

3) XGBoost Regressor

Used lag features + rolling features

Underperformed due to limited feature complexity

Could not capture sudden upward trend

* Why SARIMA Won?

✔ Captures trend
✔ Captures weekly seasonality
✔ Handles financial time series smoothly
✔ Lower error metrics

* 5. Final Forecast (Refit on Full Data)

The SARIMA model was retrained on the full dataset and used to predict the next 30–200 business days.

Outputs include:

Predicted close prices

95% confidence intervals

Business-day-based future index

* 6. Streamlit Deployment

The web app includes:

✔ Last 200 days actual prices
✔ SARIMA forecast with confidence intervals
✔ Forecast horizon slider
✔ Download forecast as CSV
✔ View SARIMA model summary
✔ Model configuration tab
✔ "Verify with Colab" section to ensure same predictions
✔ Clean and interactive UI

* 7. How to Run the App
Install dependencies:
pip install streamlit pandas numpy statsmodels matplotlib

Run app:
streamlit run app.py



* 8. Results

SARIMA produced the most accurate predictions

Confidence intervals show uncertainty increasing over long forecast horizons

Deployment allows interactive forecasting

Predictions match exactly with Colab notebook

* 9. Future Enhancements

Include LSTM / Prophet for comparison

Add macroeconomic features (S&P 500, VIX, CPI)

Deploy on Streamlit Cloud

Add real-time market API support
