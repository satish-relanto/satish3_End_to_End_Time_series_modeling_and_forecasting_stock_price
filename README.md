# Stock Price Forecasting And Risk Dashboard With ARIMA

Deployment-ready ARIMA forecasting project for stock-price research. The project trains an ARIMA model on historical OHLCV data, evaluates it with walk-forward validation, serves forecasts through FastAPI, and provides a Streamlit dashboard for visual inspection.

This is a research and learning project, not financial advice.

## What Is Included

- Local CSV ingestion for OHLCV market data.
- Log-price modelling with ARIMA.
- ACF/PACF diagnostics on log returns, not raw prices.
- ADF stationarity testing on transformed series.
- Walk-forward validation against a naive last-price baseline.
- Forecast intervals and simple risk summary.
- FastAPI endpoint for deployment.
- Streamlit dashboard for analyst inspection.
- Docker and Docker Compose setup.
- Tests for preprocessing, forecasting, and API behavior.

## Project Structure

```text
api/                  FastAPI application
case-studies/         Selected case-study document
data/raw/             Sample OHLCV data
frontend/             Streamlit dashboard
models/               Trained model artifacts
reports/              Evaluation and diagnostics outputs
src/stock_arima/      Reusable forecasting package
tests/                Automated tests
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

## Train A Model

```powershell
python -m stock_arima.train --ticker AAPL --order 1,1,1
```

This writes:

- `models/AAPL_adjusted_close_arima.joblib`
- `reports/AAPL_adjusted_close_acf_pacf.csv`
- `reports/AAPL_adjusted_close_model_metadata.json`

## Evaluate

```powershell
python -m stock_arima.evaluate --ticker AAPL --horizon-days 5
```

The report is saved under `reports/`.

## Run The API

```powershell
python -m uvicorn api.main:app --host localhost --port 8000
```

Open:

- `http://localhost:8000/docs`
- `http://localhost:8000/health`

Tip: Uvicorn prints the base server URL, `http://localhost:8000`. Opening that URL redirects to `/docs`.

Example request:

```json
{
  "ticker": "AAPL",
  "target": "adjusted_close",
  "horizon_days": 10
}
```

Endpoint:

```text
POST /forecast/stock
```

## Run The Dashboard

```powershell
python -m streamlit run frontend/app.py
```

## Docker

Build and run the API:

```powershell
docker build -t stock-arima-forecasting .
docker run --rm -p 8000:8000 stock-arima-forecasting
```

Run API and dashboard together:

```powershell
docker compose up --build
```

API:

```text
http://localhost:8000/docs
```

Dashboard:

```text
http://localhost:8501
```

## Data Format

The CSV must include:

```text
date,ticker,open,high,low,close,volume
```

`adjusted_close` is recommended. If it is absent, the loader copies `close` into `adjusted_close`.

## Environment Variables

See `.env.example`.

Key settings:

- `STOCK_ARIMA_DATA_PATH`
- `STOCK_ARIMA_MODEL_DIR`
- `STOCK_ARIMA_REPORT_DIR`
- `STOCK_ARIMA_DEFAULT_TICKER`
- `STOCK_ARIMA_TARGET`
- `STOCK_ARIMA_ORDER`
- `STOCK_ARIMA_MAX_HORIZON_DAYS`

## Tests

```powershell
python -m pytest
```

## Methodology Notes

Stock prices are usually non-stationary and often behave close to a random walk. This project uses ARIMA on log prices, while stationarity diagnostics and ACF/PACF analysis are applied to log returns or differenced log prices. That makes the diagnostics more meaningful than applying them blindly to raw prices.

The naive last-price forecast is included as a baseline because any financial forecasting model should be compared against a simple random-walk-style benchmark.
