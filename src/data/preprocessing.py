"""Reshaping and normalisation of the price panel for the autoencoder.

Design choices (from the workshop brief and correction lecture):

* **Simple** daily returns ``r_t = P_t / P_{t-1} - 1`` (not log returns),
  the same definition used by the tracking-error metric.
* One **row per stock**, its training-window daily returns as columns
  (~342 x ~1762). Each stock is a sample the encoder maps to the latent
  space; the 25 tracker stocks are the *output* of clustering, not the
  input.
* ``StandardScaler`` fitted **only on train** standardises each column
  (each day), i.e. a cross-sectional z-score, so no highly volatile day
  dominates the loss. Validation/test never go through the autoencoder,
  so there is no look-ahead leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.dataset import DateWindow
from src.utils.config import BENCHMARK_NAME

REQUIRED_PRICE_COLUMNS = ("date", "symbol", "close")


def to_wide(prices_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long price panel to wide format.

    Args:
        prices_long: Columns ``date``, ``symbol``, ``close``.

    Returns:
        Prices with index = date (sorted) and one column per ticker.

    Raises:
        ValueError: If a required column is missing.
    """
    missing = set(REQUIRED_PRICE_COLUMNS) - set(prices_long.columns)
    if missing:
        raise ValueError(f"Price panel is missing columns {sorted(missing)}.")
    wide = prices_long.pivot(index="date", columns="symbol", values="close")
    wide.index = pd.DatetimeIndex(wide.index)
    return wide.sort_index()


def benchmark_series(
    benchmark_long: pd.DataFrame, column: str
) -> pd.Series:
    """Extract the benchmark price series indexed by date.

    Args:
        benchmark_long: Table with a ``date`` column and price columns.
        column: Price column to use (``rsp_adj`` = total return).

    Returns:
        The benchmark series named ``"RSP"``.

    Raises:
        ValueError: If ``date`` or ``column`` is missing.
    """
    for col in ("date", column):
        if col not in benchmark_long.columns:
            raise ValueError(f"Benchmark table has no column '{col}'.")
    series = benchmark_long.set_index("date")[column].sort_index()
    series.index = pd.DatetimeIndex(series.index)
    series.name = BENCHMARK_NAME
    return series


def to_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Simple daily returns from adjusted prices.

    Args:
        prices: Price Series or DataFrame indexed by date.

    Returns:
        Returns with the same shape (first row is NaN).
    """
    return prices.pct_change(fill_method=None)


def asset_return_matrix(
    prices_wide: pd.DataFrame, window: DateWindow
) -> tuple[np.ndarray, list[str]]:
    """Build the autoencoder input matrix: one row per stock.

    Args:
        prices_wide: Wide price panel.
        window: Window whose returns become the features (train).

    Returns:
        ``(X, tickers)`` with ``X`` of shape ``(n_stocks, n_days)``. The
        first (NaN) return of the window is dropped.

    Raises:
        ValueError: If the window has fewer than two prices or the matrix
            contains missing values.
    """
    prices = window.slice(prices_wide)
    if len(prices) < 2:
        raise ValueError(
            f"Window '{window.name}' contains {len(prices)} price rows; "
            "at least 2 are needed to compute returns."
        )
    returns = to_returns(prices).iloc[1:]
    if returns.isna().to_numpy().any():
        bad = returns.columns[returns.isna().any()].tolist()
        raise ValueError(
            f"Missing returns in window '{window.name}' for {bad[:10]}. "
            "The universe must be a rectangular (complete) panel."
        )
    return returns.T.to_numpy(dtype=np.float64), list(returns.columns)


def fit_scaler(x_train: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    """Fit a per-day (column-wise) standard scaler on training data only.

    Args:
        x_train: Matrix ``(n_stocks, n_days)``.

    Returns:
        ``(scaler, x_scaled)``.

    Raises:
        ValueError: If ``x_train`` is not 2-D.
    """
    if x_train.ndim != 2:
        raise ValueError(f"x_train must be 2-D, got ndim={x_train.ndim}.")
    scaler = StandardScaler()
    return scaler, scaler.fit_transform(x_train)
