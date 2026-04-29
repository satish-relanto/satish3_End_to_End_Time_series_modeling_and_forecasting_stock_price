# Folder Structure And File Guide

This document explains every important folder and file in the Stock ARIMA Forecasting project.

Use this when introducing the project layout to juniors before going into the modelling details.

## Root-Level Files

### `.env.example`

Example environment variable file.

It shows configurable values such as:

- Data file path.
- Model directory.
- Report directory.
- Default ticker.
- ARIMA order.
- Maximum forecast horizon.

Developers can copy this to `.env` if they want local configuration.

### `.gitignore`

Tells Git which files and folders should not be committed.

Examples:

- `.venv/`
- Python cache files.
- Test/lint caches.
- Local logs.
- Generated model artifacts.
- Generated reports.

### `README.md`

Main project documentation.

It explains:

- What the project does.
- How to install dependencies.
- How to train the model.
- How to evaluate the model.
- How to run the API.
- How to run the dashboard.
- How to use Docker.

This is the first file a new developer should read.

### `requirements.txt`

Production/runtime dependencies.

These packages are needed to run the actual application:

- FastAPI.
- Pandas.
- NumPy.
- Statsmodels.
- Streamlit.
- Uvicorn.
- Plotly.
- Joblib.

### `requirements-dev.txt`

Development dependencies.

It includes `requirements.txt` plus tools for development and testing:

- `pytest`
- `ruff`
- `httpx`

Use this when setting up the project for development.

### `pyproject.toml`

Python project configuration file.

It defines:

- Project name.
- Version.
- Python version requirement.
- Package dependencies.
- Package discovery under `src/`.
- Pytest configuration.
- Ruff configuration.

This makes the project installable as a proper Python package.

### `Dockerfile`

Instructions for building the API container image.

It:

- Uses Python base image.
- Installs runtime dependencies.
- Copies project files.
- Starts the FastAPI app with Uvicorn.

### `docker-compose.yml`

Runs multiple services together.

Current services:

- `api`: FastAPI forecasting service.
- `dashboard`: Streamlit dashboard.

This is useful for local deployment testing.

## `api/`

Contains the FastAPI application.

### `api/__init__.py`

Marks the `api` directory as a Python package.

It does not contain business logic.

### `api/main.py`

Main API application file.

It defines:

- FastAPI app metadata.
- Request schema using Pydantic.
- Health endpoint.
- Ticker listing endpoint.
- Forecast endpoint.

Endpoints:

```text
GET /health
GET /tickers
POST /forecast/stock
```

This file connects external API requests to the internal forecasting code.

## `case-studies/`

Contains the selected case study document.

### `case-studies/03-stock-price-forecasting-risk-dashboard.md`

Original project case study.

It explains:

- Business context.
- Problem statement.
- Dataset options.
- Modelling scope.
- API design.
- Dashboard features.
- Deployment plan.
- Risks and acceptance criteria.

This was the planning document before implementation.

## `data/`

Contains project data.

### `data/raw/`

Stores raw input data.

Raw data means the source-like data before project transformations.

### `data/raw/sample_ohlcv.csv`

Sample stock market dataset.

It contains 1000 rows of synthetic AAPL-style business-day OHLCV data.

Columns:

```text
date
ticker
open
high
low
close
adjusted_close
volume
```

This file allows the whole project to run without internet access.

### `data/processed/`

Reserved for processed datasets.

Currently, preprocessing happens in memory, so this folder is mostly a placeholder.

### `data/processed/.gitkeep`

Empty placeholder file.

Git does not track empty folders, so `.gitkeep` allows the `data/processed/` folder to remain in the repo.

## `docs/`

Contains detailed documentation.

### `docs/project_explanation_for_juniors.md`

Teaching guide for explaining the full project.

It covers:

- Business problem.
- Time series concepts.
- ARIMA explanation.
- ACF/PACF.
- Stationarity.
- Code modules.
- API.
- Dashboard.
- Testing.
- Deployment.

### `docs/folder_structure_and_file_guide.md`

This file.

It explains every folder and file in the project.

Use it when onboarding someone to the repository layout.

## `frontend/`

Contains the Streamlit dashboard.

### `frontend/app.py`

Dashboard application.

It allows users to:

- Select ticker.
- Select target column.
- Choose forecast horizon.
- View historical prices.
- View ARIMA forecast.
- View prediction intervals.
- Compare against naive baseline.
- View risk metrics.
- Inspect model metadata.

This file is useful for visual explanation and demos.

Run it with:

```powershell
python -m streamlit run frontend/app.py
```

## `models/`

Stores trained model artifacts.

### `models/AAPL_adjusted_close_arima.joblib`

Serialized trained ARIMA model artifact.

It contains:

- Fitted Statsmodels ARIMA result.
- Model metadata.

The API and dashboard load this model to generate forecasts.

Generated model files are usually ignored by Git because they can become large and are reproducible.

## `reports/`

Stores generated diagnostics and evaluation outputs.

### `reports/AAPL_adjusted_close_acf_pacf.csv`

ACF/PACF diagnostic table.

It is generated from log returns.

Columns:

```text
lag
acf
pacf
```

This helps explain possible ARIMA lag structure.

### `reports/AAPL_adjusted_close_evaluation.json`

Walk-forward evaluation report.

It contains:

- Ticker.
- Target.
- ARIMA order.
- Forecast horizon.
- ARIMA metrics.
- Naive baseline metrics.
- Prediction rows.

Used to judge model performance.

### `reports/AAPL_adjusted_close_model_metadata.json`

Training metadata.

It contains:

- Ticker.
- Target.
- ARIMA order.
- Training start date.
- Training end date.
- AIC.
- BIC.
- Last observed price.
- Recent volatility.
- Stationarity test result on log returns.

This file helps with model traceability.

## `scripts/`

Contains utility scripts.

### `scripts/generate_sample_data.py`

Generates the sample stock dataset.

It creates:

- 1000 business-day rows.
- Synthetic AAPL-style OHLCV data.
- Reproducible output using a fixed random seed.

Run it with:

```powershell
python scripts/generate_sample_data.py
```

### `scripts/check_forecast.py`

Simple forecast smoke check.

It calls the forecasting code and prints a 5-day forecast.

Use it to quickly confirm:

- Model loads correctly.
- Forecasting works.
- Output format is valid.

Run it with:

```powershell
python scripts/check_forecast.py
```

## `src/`

Contains the main reusable Python source code.

The project uses a `src` layout, which is common in production Python projects.

Why use `src/`?

- Avoids accidental imports from the project root.
- Makes package installation cleaner.
- Encourages proper package structure.

## `src/stock_arima/`

Main Python package for the forecasting system.

### `src/stock_arima/__init__.py`

Marks `stock_arima` as a Python package.

Also contains the package version:

```python
__version__ = "0.1.0"
```

### `src/stock_arima/config.py`

Central configuration module.

It defines default settings and reads environment variables.

Important class:

```python
Settings
```

Important values:

- `data_path`
- `model_dir`
- `report_dir`
- `default_ticker`
- `target`
- `order`
- `max_horizon_days`

This avoids hardcoding paths throughout the project.

### `src/stock_arima/market_data.py`

Market data loading and validation.

Main functions:

```python
load_ohlcv_csv(path)
filter_ticker(df, ticker)
available_tickers(df)
```

Used by:

- Training.
- Evaluation.
- API.
- Dashboard.
- Tests.

### `src/stock_arima/preprocess.py`

Time series preprocessing.

Main functions:

```python
prepare_price_series(...)
log_price_series(...)
log_return_series(...)
stationarity_summary(...)
```

Responsibilities:

- Filter ticker.
- Set date index.
- Sort by date.
- Convert to business-day frequency.
- Forward-fill missing business days.
- Validate positive prices.
- Create log prices.
- Create log returns.
- Run ADF stationarity test.

### `src/stock_arima/diagnostics.py`

Diagnostic utilities.

Main function:

```python
acf_pacf_table(series, lags=20)
```

It creates a table of ACF and PACF values for transformed stationary series.

In this project, it is used on log returns.

### `src/stock_arima/model_store.py`

Model artifact saving and loading.

Main functions:

```python
model_filename(...)
model_path(...)
save_artifact(...)
load_artifact(...)
```

This keeps model file naming and serialization logic in one place.

### `src/stock_arima/train.py`

Model training module.

Main function:

```python
train_model(...)
```

It:

1. Loads data.
2. Prepares the price series.
3. Converts prices to log prices.
4. Calculates log returns.
5. Creates ACF/PACF diagnostics.
6. Runs stationarity test.
7. Fits ARIMA.
8. Saves the model artifact.
9. Saves training metadata.

It can also be run as a command:

```powershell
python -m stock_arima.train --ticker AAPL --order 1,1,1
```

### `src/stock_arima/forecast.py`

Forecast generation module.

Main functions:

```python
ensure_model(...)
forecast_prices(...)
```

It:

- Checks if a trained model exists.
- Trains one if needed.
- Loads the model artifact.
- Generates future log-price forecasts.
- Converts forecasts back to price scale.
- Creates business-day future dates.
- Adds confidence intervals.
- Adds naive baseline.
- Adds risk summary.

Used by:

- API.
- Dashboard.
- Forecast smoke script.
- Tests.

### `src/stock_arima/evaluate.py`

Walk-forward evaluation module.

Main functions:

```python
regression_metrics(...)
walk_forward_evaluate(...)
```

It evaluates ARIMA forecasts over multiple rolling windows.

Metrics:

- MAE.
- RMSE.
- MAPE.

It also compares ARIMA against a naive last-price baseline.

Run it with:

```powershell
python -m stock_arima.evaluate --ticker AAPL --horizon-days 3
```

## `tests/`

Contains automated tests.

### `tests/test_market_data.py`

Tests market data loading.

Checks:

- Sample CSV loads correctly.
- `AAPL` exists.
- Required columns are present.

### `tests/test_preprocess.py`

Tests preprocessing logic.

Checks:

- Price series uses business-day frequency.
- No missing values after preprocessing.
- Prices are positive.
- Log returns have expected length.

### `tests/test_forecast.py`

Tests forecast output shape.

Checks:

- Forecast returns correct ticker.
- Forecast horizon length is correct.
- Forecast rows contain required keys.

### `tests/test_api.py`

Tests API health endpoint.

Checks:

- `/health` returns HTTP 200.
- Response is `{"status": "ok"}`.

## Generated Or Local-Only Folders

### `.venv/`

Local Python virtual environment.

It stores installed packages.

This should not be committed to Git.

### `.pytest_cache/`

Pytest cache folder.

Generated when tests run.

Safe to delete.

### `.ruff_cache/`

Ruff linting cache folder.

Generated when Ruff runs.

Safe to delete.

### `__pycache__/`

Python bytecode cache folders.

Generated automatically when Python files are imported or compiled.

Safe to delete.

## End-To-End File Flow

The main runtime flow looks like this:

```text
data/raw/sample_ohlcv.csv
      |
      v
src/stock_arima/market_data.py
      |
      v
src/stock_arima/preprocess.py
      |
      v
src/stock_arima/diagnostics.py
      |
      v
src/stock_arima/train.py
      |
      v
models/AAPL_adjusted_close_arima.joblib
reports/*.csv and reports/*.json
      |
      v
src/stock_arima/forecast.py
      |
      v
api/main.py or frontend/app.py
```

## What To Show Juniors First

Recommended order:

1. `README.md`
2. `data/raw/sample_ohlcv.csv`
3. `src/stock_arima/market_data.py`
4. `src/stock_arima/preprocess.py`
5. `src/stock_arima/train.py`
6. `src/stock_arima/forecast.py`
7. `api/main.py`
8. `frontend/app.py`
9. `tests/`
10. `Dockerfile` and `docker-compose.yml`

This order moves from simple project usage to internal implementation and finally deployment.
