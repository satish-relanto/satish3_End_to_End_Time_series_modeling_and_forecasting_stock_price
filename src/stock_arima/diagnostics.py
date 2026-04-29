"""Diagnostic helpers for checking whether ARIMA inputs look well behaved."""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import acf, pacf


def acf_pacf_table(series: pd.Series, lags: int = 20) -> pd.DataFrame:
    """Build an ACF/PACF table for a stationary transformed series."""
    clean = series.dropna()
    # PACF cannot use more than about half the sample size, so cap small datasets safely.
    max_lags = min(lags, max(1, len(clean) // 2 - 1))

    # ACF/PACF values are exported as a table so analysts can inspect lag
    # structure in reports without re-running statsmodels interactively.
    acf_values = acf(clean, nlags=max_lags, fft=False)
    pacf_values = pacf(clean, nlags=max_lags, method="ywm")
    return pd.DataFrame(
        {
            "lag": range(max_lags + 1),
            "acf": acf_values,
            "pacf": pacf_values,
        }
    )
