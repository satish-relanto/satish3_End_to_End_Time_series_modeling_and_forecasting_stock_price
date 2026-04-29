"""Tests for price-series preprocessing and transformations."""

from stock_arima.config import settings
from stock_arima.market_data import load_ohlcv_csv
from stock_arima.preprocess import log_return_series, prepare_price_series


def test_prepare_price_series_returns_business_day_series():
    # ARIMA expects regular spacing, so preprocessing should return a complete
    # business-day index with usable positive prices.
    df = load_ohlcv_csv(settings.data_path)
    series = prepare_price_series(df, ticker="AAPL")

    assert series.index.freqstr == "B"
    assert series.notna().all()
    assert (series > 0).all()


def test_log_returns_are_one_observation_shorter():
    # First differencing removes the initial observation because there is no
    # previous price to compare against.
    df = load_ohlcv_csv(settings.data_path)
    series = prepare_price_series(df, ticker="AAPL")
    returns = log_return_series(series)

    assert len(returns) == len(series) - 1
    assert returns.name == "log_return"
