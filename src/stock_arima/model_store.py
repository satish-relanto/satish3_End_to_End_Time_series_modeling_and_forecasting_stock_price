"""Persistence helpers for trained statsmodels artifacts and metadata bundles."""

from __future__ import annotations

from pathlib import Path

import joblib


def model_filename(ticker: str, target: str) -> str:
    """Create a filesystem-safe artifact filename for a ticker/target pair."""
    # Tickers can contain exchange separators; replacing slashes avoids creating
    # accidental subdirectories under the model folder.
    safe_ticker = ticker.upper().replace("/", "_")
    safe_target = target.lower().replace("/", "_")
    return f"{safe_ticker}_{safe_target}_arima.joblib"


def model_path(model_dir: str | Path, ticker: str, target: str) -> Path:
    """Resolve where a model artifact should live."""
    return Path(model_dir) / model_filename(ticker, target)


def save_artifact(artifact: dict, path: str | Path) -> Path:
    """Persist a trained model bundle and create its parent directory if needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return output_path


def load_artifact(path: str | Path) -> dict:
    """Load a saved model bundle, failing clearly when the file is absent."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    return joblib.load(artifact_path)
