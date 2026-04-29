"""Market-data loading and normalization utilities for local OHLCV CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# The rest of the pipeline assumes these canonical lower-case column names.
REQUIRED_COLUMNS = {"date", "ticker", "open", "high", "low", "close", "volume"}


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load market data from a local OHLCV CSV and normalize its schema."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Market data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Normalize headers at ingestion so callers can provide human-edited CSVs
    # with harmless casing or whitespace differences.
    df.columns = [column.strip().lower() for column in df.columns]

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Market data is missing required columns: {sorted(missing)}")

    # Some public datasets omit adjusted close; falling back to close keeps the
    # downstream target name stable for examples and tests.
    if "adjusted_close" not in df.columns:
        df["adjusted_close"] = df["close"]

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    numeric_columns = ["open", "high", "low", "close", "adjusted_close", "volume"]
    for column in numeric_columns:
        # Fail early on malformed prices/volume instead of carrying object
        # dtype values into modeling code.
        df[column] = pd.to_numeric(df[column], errors="raise")

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def filter_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a single ticker's rows, with a helpful error for unknown symbols."""
    ticker = ticker.upper().strip()
    ticker_df = df.loc[df["ticker"] == ticker].copy()
    if ticker_df.empty:
        available = ", ".join(sorted(df["ticker"].unique()))
        raise ValueError(f"Ticker '{ticker}' was not found. Available tickers: {available}")
    return ticker_df.sort_values("date").reset_index(drop=True)


def available_tickers(df: pd.DataFrame) -> list[str]:
    """List normalized ticker symbols present in a loaded market-data frame."""
    return sorted(df["ticker"].dropna().astype(str).str.upper().unique().tolist())
