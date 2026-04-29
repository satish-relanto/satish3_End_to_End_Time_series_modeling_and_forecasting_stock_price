"""Training workflow for fitting, evaluating, and persisting ARIMA artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from stock_arima.config import settings
from stock_arima.diagnostics import acf_pacf_table
from stock_arima.market_data import load_ohlcv_csv
from stock_arima.model_store import model_path, save_artifact
from stock_arima.preprocess import log_price_series, log_return_series, prepare_price_series, stationarity_summary


def train_model(
    data_path: str | Path = settings.data_path,
    model_dir: str | Path = settings.model_dir,
    report_dir: str | Path = settings.report_dir,
    ticker: str = settings.default_ticker,
    target: str = settings.target,
    order: tuple[int, int, int] = settings.order,
) -> Path:
    """Train an ARIMA model, save diagnostics, and return the artifact path."""
    # Training always starts from raw CSV input so saved artifacts can be
    # reproduced from the configured data path.
    df = load_ohlcv_csv(data_path)
    price_series = prepare_price_series(df, ticker=ticker, target=target)
    log_series = log_price_series(price_series)
    returns = log_return_series(price_series)

    # Model log prices so forecasts remain positive after converting back with exp().
    model = ARIMA(log_series, order=order, enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit()

    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    # ACF/PACF diagnostics help validate whether the configured p and q terms are reasonable.
    diagnostics = acf_pacf_table(returns)
    diagnostics.to_csv(report_path / f"{ticker.upper()}_{target}_acf_pacf.csv", index=False)

    stationarity = stationarity_summary(returns)
    # Keep model metrics beside the artifact so the API can explain its outputs.
    metadata = {
        "ticker": ticker.upper(),
        "target": target,
        "order": order,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_start": price_series.index.min().date().isoformat(),
        "training_end": price_series.index.max().date().isoformat(),
        "last_observed_price": float(price_series.iloc[-1]),
        "nobs": int(result.nobs),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "recent_volatility": float(returns.tail(30).std()),
        "stationarity_on_log_returns": stationarity,
    }

    output_path = model_path(model_dir, ticker=ticker, target=target)
    save_artifact({"model": result, "metadata": metadata}, output_path)
    pd.Series(metadata).to_json(report_path / f"{ticker.upper()}_{target}_model_metadata.json", indent=2)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line options for one-off model training."""
    parser = argparse.ArgumentParser(description="Train an ARIMA stock forecasting model.")
    parser.add_argument("--data-path", default=str(settings.data_path))
    parser.add_argument("--model-dir", default=str(settings.model_dir))
    parser.add_argument("--report-dir", default=str(settings.report_dir))
    parser.add_argument("--ticker", default=settings.default_ticker)
    parser.add_argument("--target", default=settings.target)
    parser.add_argument("--order", default=",".join(map(str, settings.order)))
    return parser.parse_args()


def main() -> None:
    """CLI entry point used by local scripts and container commands."""
    args = parse_args()
    # argparse receives the ARIMA order as text; convert it at the CLI boundary
    # so train_model can stay typed as a tuple.
    order = tuple(int(part) for part in args.order.split(","))
    output_path = train_model(
        data_path=args.data_path,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        ticker=args.ticker,
        target=args.target,
        order=order,  # type: ignore[arg-type]
    )
    print(f"Saved model artifact to {output_path}")


if __name__ == "__main__":
    main()
