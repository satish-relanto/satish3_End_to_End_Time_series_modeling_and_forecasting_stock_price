"""Configuration defaults and environment-variable parsing for the project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# Resolve paths from the installed source file so commands work from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_order(value: str) -> tuple[int, int, int]:
    """Parse an ARIMA order from an environment-friendly p,d,q string."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("ARIMA order must use the format p,d,q, for example 1,1,1.")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


@dataclass(frozen=True)
class Settings:
    """Centralized runtime settings with environment-variable overrides."""

    # Defaults target the checked-in sample data and local artifact folders;
    # deployments can override each value with STOCK_ARIMA_* environment vars.
    data_path: Path = Path(os.getenv("STOCK_ARIMA_DATA_PATH", PROJECT_ROOT / "data" / "raw" / "sample_ohlcv.csv"))
    model_dir: Path = Path(os.getenv("STOCK_ARIMA_MODEL_DIR", PROJECT_ROOT / "models"))
    report_dir: Path = Path(os.getenv("STOCK_ARIMA_REPORT_DIR", PROJECT_ROOT / "reports"))
    default_ticker: str = os.getenv("STOCK_ARIMA_DEFAULT_TICKER", "AAPL").upper()
    target: str = os.getenv("STOCK_ARIMA_TARGET", "adjusted_close")
    order: tuple[int, int, int] = parse_order(os.getenv("STOCK_ARIMA_ORDER", "1,1,1"))
    max_horizon_days: int = int(os.getenv("STOCK_ARIMA_MAX_HORIZON_DAYS", "30"))


settings = Settings()
