"""Walk-forward evaluation utilities for measuring forecasting accuracy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from stock_arima.config import settings
from stock_arima.market_data import load_ohlcv_csv
from stock_arima.preprocess import log_price_series, prepare_price_series


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute common scale and percentage error metrics for forecasts."""
    # Keep metric definitions together so ARIMA and baseline results are
    # calculated identically.
    errors = actual - predicted
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": float(np.mean(np.abs(errors / actual)) * 100),
    }


def walk_forward_evaluate(
    data_path: str | Path = settings.data_path,
    ticker: str = settings.default_ticker,
    target: str = settings.target,
    order: tuple[int, int, int] = settings.order,
    horizon_days: int = 5,
    initial_train_size: int = 45,
    step_size: int = 5,
) -> dict:
    """Run rolling-origin evaluation to mimic repeated future forecasting."""
    # Require enough observations for the initial fit plus at least one complete
    # forecast horizon.
    df = load_ohlcv_csv(data_path)
    price_series = prepare_price_series(df, ticker=ticker, target=target, min_observations=initial_train_size + horizon_days)
    log_series = log_price_series(price_series)

    predictions: list[float] = []
    actuals: list[float] = []
    rows: list[dict] = []

    for train_end in range(initial_train_size, len(log_series) - horizon_days + 1, step_size):
        # Each fold trains only on data available up to train_end, then forecasts
        # the next horizon to avoid leaking future observations.
        train = log_series.iloc[:train_end]
        test_prices = price_series.iloc[train_end : train_end + horizon_days]
        result = ARIMA(train, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
        forecast_log = result.forecast(steps=horizon_days)
        forecast_prices = np.exp(forecast_log.to_numpy())

        predictions.extend(forecast_prices.tolist())
        actuals.extend(test_prices.to_numpy().tolist())
        for date, actual, predicted in zip(test_prices.index, test_prices.to_numpy(), forecast_prices):
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "actual": float(actual),
                    "predicted": float(predicted),
                    "naive_baseline": float(price_series.iloc[train_end - 1]),
                }
            )

    if not rows:
        raise ValueError("Not enough observations to run walk-forward evaluation with the chosen settings.")

    metrics = regression_metrics(np.array(actuals), np.array(predictions))
    # The last-observed-price baseline gives the ARIMA results a simple benchmark.
    naive_metrics = regression_metrics(
        np.array([row["actual"] for row in rows]),
        np.array([row["naive_baseline"] for row in rows]),
    )
    return {
        "ticker": ticker.upper(),
        "target": target,
        "order": order,
        "horizon_days": horizon_days,
        "metrics": metrics,
        "naive_baseline_metrics": naive_metrics,
        "predictions": rows,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options for walk-forward evaluation."""
    parser = argparse.ArgumentParser(description="Run walk-forward ARIMA evaluation.")
    parser.add_argument("--data-path", default=str(settings.data_path))
    parser.add_argument("--report-dir", default=str(settings.report_dir))
    parser.add_argument("--ticker", default=settings.default_ticker)
    parser.add_argument("--target", default=settings.target)
    parser.add_argument("--order", default=",".join(map(str, settings.order)))
    parser.add_argument("--horizon-days", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """CLI entry point that writes the evaluation report JSON."""
    args = parse_args()
    order = tuple(int(part) for part in args.order.split(","))
    report = walk_forward_evaluate(
        data_path=args.data_path,
        ticker=args.ticker,
        target=args.target,
        order=order,  # type: ignore[arg-type]
        horizon_days=args.horizon_days,
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / f"{args.ticker.upper()}_{args.target}_evaluation.json"
    pd.Series(report).to_json(output_path, indent=2)
    print(f"Saved evaluation report to {output_path}")
    print(report["metrics"])


if __name__ == "__main__":
    main()
