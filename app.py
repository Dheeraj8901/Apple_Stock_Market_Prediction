# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import statsmodels.api as sm
from io import BytesIO

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Apple Stock Forecast (SARIMA)",
    layout="wide",
)

# ---------------------------
# CUSTOM CSS FOR BEAUTIFUL UI
# ---------------------------
st.markdown("""
<style>

html, body {
    font-family: 'Roboto', sans-serif;
}

.big-title {
    font-size: 42px !important;
    font-weight: 800 !important;
    background: -webkit-linear-gradient(90deg, #ff4b4b, #0066ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-card {
    padding: 18px;
    border-radius: 12px;
    color: white;
    font-weight: 700;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
}

.card-blue { background: linear-gradient(135deg, #4e73df, #224abe); }
.card-green { background: linear-gradient(135deg, #1cc88a, #13855c); }
.card-orange { background: linear-gradient(135deg, #f6ad55, #dd6b20); }
.card-purple { background: linear-gradient(135deg, #9f7aea, #6b46c1); }

.sidebar .sidebar-content {
    background-color: #f8f9fc;
}

table {
    border-radius: 10px !important;
    overflow: hidden !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helper: Load & preprocess
# ---------------------------
@st.cache_data(show_spinner=False)
def load_and_preprocess(csv_path="Stock Market.csv"):
    df = pd.read_csv(csv_path, header=0)

    # Rename if needed
    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)

    # Parse date and set index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.sort_values("Date")
        df = df.dropna(subset=["Date"])
        df.set_index("Date", inplace=True)
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df.sort_index(inplace=True)
        df = df[~df.index.isna()]

    # Feature calculations
    df["Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # IQR capping
    df_cap = df.copy()
    for col in ["Return", "Log_Return"]:
        Q1 = df_cap[col].quantile(0.25)
        Q3 = df_cap[col].quantile(0.75)
        IQR = Q3 - Q1
        low = Q1 - 1.5 * IQR
        high = Q3 + 1.5 * IQR
        df_cap[col] = df_cap[col].clip(low, high)

    # Target series y (Close)
    y = df_cap["Close"].astype(float)

    # Reindex to business days and forward fill
    full_range = pd.bdate_range(start=y.index.min(), end=y.index.max())
    y = y.reindex(full_range).ffill()
    if y.index.freq is None:
        y.index.freq = "B"

    return df_cap, y

# ---------------------------
# Load SARIMA model
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_pickled_model(pkl_path="sarima_model.pkl"):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

# ---------------------------
# UI Title
# ---------------------------
st.markdown("<h1 class='big-title'>Apple Stock Price Forecast — SARIMA</h1>", unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("⚙ Forecast Controls")
forecast_days = st.sidebar.slider(
    "Select forecast horizon (business days):",
    min_value=5, max_value=240,
    value=30, step=5
)
show_ci = st.sidebar.checkbox("Show 95% confidence interval", True)

# ---------------------------
# Load data + model
# ---------------------------
with st.spinner("Loading data & model..."):
    df_cap, y = load_and_preprocess()
    try:
        sarima_results = load_pickled_model()
    except FileNotFoundError:
        st.error(" Missing file: 'sarima_model.pkl'. Please place it near app.py.")
        st.stop()

# ---------------------------
# Top Metrics (Styled Cards)
# ---------------------------
last_date = y.index.max()
last_price = y.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"<div class='metric-card card-blue'> Last Date<br><span style='font-size:22px'>{last_date.strftime('%Y-%m-%d')}</span></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric-card card-green'> Last Close<br><span style='font-size:22px'>${last_price:.2f}</span></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric-card card-orange'> Forecast Days<br><span style='font-size:22px'>{forecast_days}</span></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='metric-card card-purple'> Observations<br><span style='font-size:22px'>{len(y)}</span></div>", unsafe_allow_html=True)

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs([" Historical Data", " Forecast", " Model Info"])

# ---------------------------
# TAB 1 — Historical Data
# ---------------------------
with tab1:
    st.subheader(" Last 200 Business Days — Apple Close Price")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y.index[-200:], y.values[-200:], linewidth=1.5)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ---------------------------
# TAB 2 — Forecast
# ---------------------------
with tab2:
    st.subheader(" Forecast Results")

    # Forecast
    forecast_obj = sarima_results.get_forecast(steps=forecast_days)
    pred_mean = forecast_obj.predicted_mean
    future_index = pd.date_range(
        start=y.index.max() + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="B"
    )
    pred_series = pd.Series(pred_mean.values, index=future_index)

    df_forecast = pred_series.to_frame("Predicted_Close")

    if show_ci:
        conf = forecast_obj.conf_int()
        df_forecast["Lower_95"] = conf.iloc[:, 0].values
        df_forecast["Upper_95"] = conf.iloc[:, 1].values

    # Table
    st.dataframe(df_forecast.style.format("{:.2f}"))

    # Download
    st.download_button(
        label=" Download Forecast CSV",
        data=df_forecast.to_csv().encode("utf-8"),
        file_name="forecast.csv"
    )

    # Plot
    st.subheader(" Visualization")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(y.index[-200:], y.values[-200:], label="Actual", linewidth=1.5)
    ax2.plot(pred_series.index, pred_series.values, label="Forecast", linewidth=1.8)

    if show_ci:
        ax2.fill_between(
            df_forecast.index,
            df_forecast["Lower_95"],
            df_forecast["Upper_95"],
            color="orange", alpha=0.2,
            label="Confidence Interval"
        )

    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

# ---------------------------
# TAB 3 — Model Info
# ---------------------------
with tab3:
    st.subheader(" SARIMA Model Summary")
    st.text_area("Summary", sarima_results.summary().as_text(), height=350)

    st.markdown("###  Model Configuration")
    st.write("Order (p,d,q): **(1,1,1)**")
    st.write("Seasonal Order (P,D,Q,s): **(1,1,1,5)**")
    st.write(f"Observations used: **{sarima_results.nobs}**")
