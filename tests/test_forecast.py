"""Tests for the high-level forecast service contract."""

from pathlib import Path

from stock_arima.config import settings
from stock_arima.forecast import forecast_prices


def test_forecast_prices_returns_expected_shape(tmp_path: Path):
    # Use a temporary model directory so the test can train/cache artifacts
    # without depending on or modifying the shared models folder.
    result = forecast_prices(
        ticker="AAPL",
        horizon_days=3,
        data_path=settings.data_path,
        model_dir=tmp_path,
        report_dir=tmp_path,
        max_horizon_days=10,
    )

    assert result["ticker"] == "AAPL"
    assert result["horizon_days"] == 3
    assert len(result["forecast"]) == 3
    assert {"date", "yhat", "lower", "upper", "naive_baseline"}.issubset(result["forecast"][0])
