"""Manual smoke script for checking forecast output from the command line."""

from __future__ import annotations

from stock_arima.forecast import forecast_prices


if __name__ == "__main__":
    # Tiny smoke check for humans: prints the forecast rows without starting the API.
    result = forecast_prices(horizon_days=5)
    for row in result["forecast"]:
        print(row)
