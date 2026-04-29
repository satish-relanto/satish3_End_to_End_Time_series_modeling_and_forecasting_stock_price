# Case Study 3: Stock Price Forecasting And Risk Dashboard With ARIMA

## Business Context

An investment research team wants a lightweight forecasting and risk-monitoring tool for selected stock prices. The tool should help analysts compare recent price trends, short-term forecasts, residual behavior, and uncertainty.

## Problem Statement

Build an end-to-end ARIMA-based forecasting system that downloads or ingests historical stock price data, trains forecasting models, and serves forecasts through an API and dashboard.

## Target Users

- Financial analysts.
- Quant research learners.
- Portfolio monitoring teams.

## Dataset Options

- Yahoo Finance historical OHLCV data through `yfinance`.
- Stooq daily market data.
- A custom CSV with columns such as `date`, `ticker`, `open`, `high`, `low`, `close`, `volume`.

## Forecasting Goal

Forecast adjusted closing price or log returns for the next 5, 10, or 30 trading days.

## Modelling Scope

- Historical price ingestion by ticker.
- Trading-calendar-aware preprocessing.
- Missing market day handling.
- Log transformation or return calculation.
- Stationarity testing.
- ACF/PACF analysis on transformed stationary series such as log returns or differenced log prices.
- ARIMA on log prices or returns.
- Optional ARIMAX using market index returns or volume.
- Walk-forward validation.
- Residual diagnostics.
- Risk summary using forecast intervals and recent volatility.

## Evaluation Metrics

- MAE.
- RMSE.
- MAPE for price forecasts.
- Directional accuracy.
- Forecast interval coverage.
- Baseline comparison against naive last-value forecast.

## End-to-End Project Components

- `data/`: raw and cached market data.
- `notebooks/`: EDA, transformation experiments, diagnostics.
- `src/market_data.py`: data loading or download.
- `src/preprocess.py`: return calculation and stationarity preparation.
- `src/train.py`: ARIMA model training.
- `src/evaluate.py`: walk-forward validation.
- `src/forecast.py`: forecast generation and interval formatting.
- `api/`: FastAPI service.
- `frontend/`: Streamlit risk dashboard.
- `models/`: per-ticker serialized model artifacts.
- `tests/`: tests for market data schema, preprocessing, and API responses.
- Docker deployment files.

## API Design

### `POST /forecast/stock`

Request:

```json
{
  "ticker": "AAPL",
  "target": "adjusted_close",
  "horizon_days": 10
}
```

Response:

```json
{
  "ticker": "AAPL",
  "target": "adjusted_close",
  "forecast": [
    {
      "date": "2026-05-01",
      "yhat": 192.45,
      "lower": 184.8,
      "upper": 200.9
    }
  ],
  "risk_summary": {
    "recent_volatility": 0.018,
    "forecast_range_pct": 8.36
  },
  "model_version": "2026-04-28"
}
```

## Dashboard Features

- Enter or select a ticker.
- Select target and horizon.
- View price history and forecast interval.
- Compare ARIMA forecast against naive baseline.
- Show residual plot and ACF/PACF diagnostics for transformed series.
- Show simple risk metrics such as recent volatility and forecast range.

## Deployment Plan

- Provide local CSV ingestion so the project can run without external APIs.
- Optionally support live downloads through a configurable provider.
- Train and save per-ticker models.
- Serve forecasts with FastAPI.
- Add Streamlit dashboard for analyst use.
- Package with Docker.
- Include a `.env.example` for optional API/data provider settings.

## Risks And Considerations

- Stock prices are noisy and often close to random-walk behavior.
- Forecasts should be framed as research estimates, not financial advice.
- ARIMA may perform best as a baseline rather than a complete trading model.
- Live data providers can introduce rate limits or connectivity issues.

## Acceptance Criteria

- The project can ingest historical OHLCV data from CSV.
- At least one ticker can be trained, evaluated, and served.
- Forecast response includes uncertainty intervals and baseline comparison.
- Dashboard clearly separates historical data, forecast, and diagnostics.
- Documentation includes a financial-use disclaimer.
