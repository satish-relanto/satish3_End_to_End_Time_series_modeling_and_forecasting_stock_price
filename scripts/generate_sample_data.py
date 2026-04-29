"""Generate deterministic demo OHLCV data for local development and tests."""

from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "sample_ohlcv.csv"


def business_days(start: date, rows: int) -> list[date]:
    """Return consecutive weekday dates, matching the trading-calendar shape needed here."""
    days: list[date] = []
    current = start
    while len(days) < rows:
        # Weekends are skipped so downstream preprocessing sees business-day
        # spacing similar to market data.
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def generate_rows(rows: int = 1000) -> list[dict[str, str | int]]:
    """Create deterministic sample OHLCV rows with trend, seasonality, and noise."""
    random.seed(42)
    dates = business_days(date(2022, 1, 3), rows)
    price = 172.0
    output: list[dict[str, str | int]] = []

    for index, current_date in enumerate(dates):
        # These terms make the fake data realistic enough for demos without
        # pretending to be a market simulator.
        drift = 0.00035
        seasonal = 0.003 * math.sin(index / 28.0) + 0.0015 * math.cos(index / 73.0)
        shock = random.gauss(0, 0.014)
        overnight_gap = random.gauss(0, 0.006)

        open_price = price * (1 + overnight_gap)
        close_price = open_price * (1 + drift + seasonal + shock)
        high_price = max(open_price, close_price) * (1 + abs(random.gauss(0.006, 0.003)))
        low_price = min(open_price, close_price) * (1 - abs(random.gauss(0.006, 0.003)))

        volume_wave = 1 + 0.18 * math.sin(index / 19.0)
        volume_noise = random.randint(-8_000_000, 8_000_000)
        volume = max(18_000_000, int(64_000_000 * volume_wave + volume_noise))

        price = close_price
        adjusted_close = close_price

        output.append(
            {
                "date": current_date.isoformat(),
                "ticker": "AAPL",
                "open": f"{open_price:.2f}",
                "high": f"{high_price:.2f}",
                "low": f"{low_price:.2f}",
                "close": f"{close_price:.2f}",
                "adjusted_close": f"{adjusted_close:.2f}",
                "volume": volume,
            }
        )

    return output


def main() -> None:
    """Write the sample CSV consumed by local training, tests, API, and dashboard."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows()
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["date", "ticker", "open", "high", "low", "close", "adjusted_close", "volume"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
