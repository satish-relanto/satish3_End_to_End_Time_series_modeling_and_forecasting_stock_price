"""Tests for loading and normalizing local market-data CSV files."""

from stock_arima.config import settings
from stock_arima.market_data import available_tickers, load_ohlcv_csv


def test_load_sample_market_data_has_aapl():
    # The sample CSV is the fixture that powers local demos and forecast tests.
    df = load_ohlcv_csv(settings.data_path)

    assert "AAPL" in available_tickers(df)
    assert {"date", "ticker", "adjusted_close"}.issubset(df.columns)
