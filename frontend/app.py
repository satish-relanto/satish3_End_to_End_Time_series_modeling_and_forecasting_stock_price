"""Streamlit dashboard for exploring stock forecasts and model risk context."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_arima.config import settings
from stock_arima.forecast import forecast_prices
from stock_arima.market_data import available_tickers, load_ohlcv_csv


st.set_page_config(page_title="Stock ARIMA Forecasting", layout="wide")

st.title("Stock ARIMA Forecasting And Risk Dashboard")

# The dashboard reads from the same configured CSV/model paths as the API, so
# local environment overrides behave consistently across both entry points.
data = load_ohlcv_csv(settings.data_path)
tickers = available_tickers(data)

with st.sidebar:
    ticker = st.selectbox("Ticker", tickers, index=tickers.index(settings.default_ticker) if settings.default_ticker in tickers else 0)
    target = st.selectbox("Target", ["adjusted_close", "close"], index=0)
    horizon = st.slider("Forecast horizon", min_value=1, max_value=settings.max_horizon_days, value=10)

forecast = forecast_prices(ticker=ticker, target=target, horizon_days=horizon)
history = data.loc[data["ticker"] == ticker].copy()
history["date"] = pd.to_datetime(history["date"])
forecast_df = pd.DataFrame(forecast["forecast"])
forecast_df["date"] = pd.to_datetime(forecast_df["date"])

# Build one combined figure so historical data, forecasts, intervals, and the
# naive baseline can be compared on the same date and price axes.
fig = go.Figure()
fig.add_trace(go.Scatter(x=history["date"], y=history[target], mode="lines", name="Historical"))
fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["yhat"], mode="lines+markers", name="ARIMA forecast"))
# Plotly fills from the lower interval trace up to this invisible upper trace.
fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["upper"], mode="lines", line={"width": 0}, showlegend=False))
fig.add_trace(
    go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["lower"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(37, 99, 235, 0.16)",
        line={"width": 0},
        name="95% interval",
    )
)
fig.add_trace(
    go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["naive_baseline"],
        mode="lines",
        name="Naive baseline",
        line={"dash": "dot"},
    )
)
fig.update_layout(height=560, margin={"l": 20, "r": 20, "t": 30, "b": 20}, xaxis_title="Date", yaxis_title=target)
st.plotly_chart(fig, width="stretch")

col1, col2, col3 = st.columns(3)
col1.metric("Recent volatility", forecast["risk_summary"]["recent_volatility"])
col2.metric("Forecast range", f'{forecast["risk_summary"]["forecast_range_pct"]}%')
col3.metric("ARIMA order", str(tuple(forecast["model_metadata"]["order"])))

with st.expander("Model metadata", expanded=False):
    st.json(forecast["model_metadata"])

st.caption("Research tool only. Forecasts are not financial advice.")
