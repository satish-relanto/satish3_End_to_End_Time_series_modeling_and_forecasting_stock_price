"""FastAPI routes that expose ticker discovery, health checks, and forecasts."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from stock_arima.config import settings
from stock_arima.forecast import forecast_prices
from stock_arima.market_data import available_tickers, load_ohlcv_csv


app = FastAPI(
    title="Stock ARIMA Forecasting API",
    description="ARIMA-based stock price forecasting service for research and diagnostics.",
    version="0.1.0",
)


class ForecastRequest(BaseModel):
    """Validated request body for the stock forecast endpoint."""

    ticker: str = Field(default=settings.default_ticker, examples=["AAPL"])
    target: str = Field(default=settings.target, examples=["adjusted_close"])
    horizon_days: int = Field(default=10, ge=1, le=settings.max_horizon_days)


@app.get("/", include_in_schema=False)
def docs_redirect() -> RedirectResponse:
    """Send browser visitors from the server root to Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight readiness check used by tests, containers, and monitors."""
    return {"status": "ok"}


@app.get("/tickers")
def tickers() -> dict[str, list[str]]:
    """Return ticker symbols available in the configured local market-data file."""
    df = load_ohlcv_csv(settings.data_path)
    return {"tickers": available_tickers(df)}


@app.post("/forecast/stock")
def forecast_stock(request: ForecastRequest) -> dict:
    """Generate an ARIMA forecast and translate domain errors into API responses."""
    try:
        # The forecasting layer owns model loading/training; the API only maps
        # validated HTTP input onto that service contract.
        return forecast_prices(
            ticker=request.ticker,
            target=request.target,
            horizon_days=request.horizon_days,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
