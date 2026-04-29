"""High-level forecasting service used by the API, dashboard, and scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stock_arima.config import settings
from stock_arima.model_store import load_artifact, model_path
from stock_arima.train import train_model


def ensure_model(
    ticker: str,
    target: str = settings.target,
    data_path: str | Path = settings.data_path,
    model_dir: str | Path = settings.model_dir,
    report_dir: str | Path = settings.report_dir,
) -> Path:
    """Return an existing model artifact path, training one if it is missing."""
    path = model_path(model_dir, ticker=ticker, target=target)
    if not path.exists():
        # Lazy training keeps first-run demos simple while still reusing saved
        # artifacts on later forecast requests.
        return train_model(data_path=data_path, model_dir=model_dir, report_dir=report_dir, ticker=ticker, target=target)
    return path


def forecast_prices(
    ticker: str = settings.default_ticker,
    target: str = settings.target,
    horizon_days: int = 10,
    model_dir: str | Path = settings.model_dir,
    report_dir: str | Path = settings.report_dir,
    data_path: str | Path = settings.data_path,
    max_horizon_days: int = settings.max_horizon_days,
) -> dict:
    """Forecast future prices and include interval and baseline risk context."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1.")
    if horizon_days > max_horizon_days:
        raise ValueError(f"horizon_days cannot exceed {max_horizon_days}.")

    artifact_path = ensure_model(
        ticker=ticker,
        target=target,
        data_path=data_path,
        model_dir=model_dir,
        report_dir=report_dir,
    )
    artifact = load_artifact(artifact_path)
    model_result = artifact["model"]
    metadata = artifact["metadata"]

    # statsmodels returns predictions on the same scale the model was trained
    # on; this project trains on log prices and converts back below.
    forecast_result = model_result.get_forecast(steps=horizon_days)
    predicted_log = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)

    # The ARIMA model is fitted on log prices, so predictions and confidence
    # bounds are exponentiated back onto the original price scale.
    lower_column, upper_column = conf_int.columns[:2]
    yhat = np.exp(predicted_log.to_numpy())
    lower = np.exp(conf_int[lower_column].to_numpy())
    upper = np.exp(conf_int[upper_column].to_numpy())

    # Forecast dates begin on the next business day after the training window.
    start_date = pd.Timestamp(metadata["training_end"]) + pd.offsets.BDay(1)
    future_dates = pd.bdate_range(start=start_date, periods=horizon_days)
    last_price = float(metadata["last_observed_price"])

    rows = []
    for date, point, low, high in zip(future_dates, yhat, lower, upper):
        rows.append(
            {
                "date": date.date().isoformat(),
                "yhat": round(float(point), 4),
                "lower": round(float(low), 4),
                "upper": round(float(high), 4),
                "naive_baseline": round(last_price, 4),
            }
        )

    # A simple range statistic gives the UI/API a quick uncertainty indicator.
    forecast_range_pct = ((max(upper) - min(lower)) / last_price) * 100
    return {
        "ticker": metadata["ticker"],
        "target": metadata["target"],
        "horizon_days": horizon_days,
        "forecast": rows,
        "risk_summary": {
            "recent_volatility": round(float(metadata["recent_volatility"]), 6),
            "forecast_range_pct": round(float(forecast_range_pct), 4),
        },
        "model_metadata": metadata,
    }
