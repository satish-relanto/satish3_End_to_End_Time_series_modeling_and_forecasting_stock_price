"""Transform raw market rows into model-ready time series and diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from stock_arima.market_data import filter_ticker


def prepare_price_series(
    df: pd.DataFrame,
    ticker: str,
    target: str = "adjusted_close",
    min_observations: int = 30,
) -> pd.Series:
    """Return a clean business-day price series for a ticker."""
    ticker_df = filter_ticker(df, ticker)
    if target not in ticker_df.columns:
        raise ValueError(f"Target column '{target}' is not present in the market data.")

    series = ticker_df.set_index("date")[target].sort_index().astype(float)
    series = series[~series.index.duplicated(keep="last")]
    # ARIMA expects an evenly spaced time index; forward filling covers market
    # holidays or missing weekdays in small local datasets.
    series = series.asfreq("B").ffill()
    series.name = target

    if len(series.dropna()) < min_observations:
        raise ValueError(
            f"Ticker '{ticker}' has {len(series.dropna())} observations after preprocessing; "
            f"at least {min_observations} are required."
        )
    if (series <= 0).any():
        raise ValueError("Price series contains non-positive values; log transformation is not valid.")
    return series


def log_price_series(price_series: pd.Series) -> pd.Series:
    """Convert positive prices to log prices before ARIMA modeling."""
    # Log prices turn multiplicative market moves into additive changes, which
    # is a better fit for the linear assumptions behind ARIMA.
    log_series = np.log(price_series.astype(float))
    log_series.name = f"log_{price_series.name or 'price'}"
    return log_series


def log_return_series(price_series: pd.Series) -> pd.Series:
    """Return first differences of log prices, commonly used as log returns."""
    returns = log_price_series(price_series).diff().dropna()
    returns.name = "log_return"
    return returns


def stationarity_summary(series: pd.Series) -> dict[str, float | int | bool]:
    """Run the Augmented Dickey-Fuller test and package the key result fields."""
    clean = series.dropna()
    if len(clean) < 10:
        raise ValueError("At least 10 observations are required for stationarity testing.")

    statistic, p_value, used_lag, nobs, *_ = adfuller(clean, autolag="AIC")
    # A p-value below 5% rejects the unit-root null in the common quick-read
    # interpretation used by this project.
    return {
        "adf_statistic": float(statistic),
        "p_value": float(p_value),
        "used_lag": int(used_lag),
        "nobs": int(nobs),
        "is_stationary_5pct": bool(p_value < 0.05),
    }
