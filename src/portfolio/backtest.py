"""Portfolio returns with quarterly rebalancing (same rule as RSP/server).

Within each calendar quarter the portfolio is buy-and-hold (weights drift
with prices); on the first trading day of the next quarter it is reset to
the target weights. This is the exact definition used by the server's
``core/scoring.quarterly_portfolio_returns``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.data.preprocessing import to_returns

WEIGHT_SUM_TOLERANCE = 1e-6


def validate_weights(weights: Mapping[str, float]) -> None:
    """Check that portfolio weights are well formed.

    Args:
        weights: Mapping ticker -> weight.

    Raises:
        ValueError: If empty, negative or not summing to 1.
    """
    if not weights:
        raise ValueError("Portfolio has no assets.")
    values = np.fromiter(weights.values(), dtype=float)
    if (values < 0).any():
        raise ValueError("Weights must be non-negative (long-only).")
    if abs(values.sum() - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"Weights must sum to 1, got {values.sum():.8f}.")


def quarterly_rebalanced_returns(
    returns: pd.DataFrame, weights: np.ndarray
) -> pd.Series:
    """Daily returns of a portfolio rebalanced every calendar quarter.

    Args:
        returns: Daily asset returns (columns aligned with ``weights``).
        weights: Target weights applied at the start of each quarter.

    Returns:
        Daily portfolio returns indexed like ``returns``.

    Raises:
        ValueError: If the number of weights and columns differ.
    """
    if returns.shape[1] != len(weights):
        raise ValueError(
            f"{returns.shape[1]} return columns but {len(weights)} weights."
        )
    out = pd.Series(index=returns.index, dtype=float)
    quarters = returns.index.to_period("Q")
    for period in pd.unique(quarters):
        mask = np.asarray(quarters == period)
        sub = returns.iloc[mask].fillna(0.0)
        # Portfolio value path inside the quarter, starting at 1.
        value = (1.0 + sub).cumprod().mul(weights, axis=1).sum(axis=1)
        prev = value.shift(1)
        prev.iloc[0] = 1.0  # quarter start = rebalance to target weights
        out.iloc[mask] = (value / prev - 1.0).to_numpy()
    return out


def portfolio_returns(
    prices_wide: pd.DataFrame, weights: Mapping[str, float]
) -> pd.Series:
    """Daily tracker returns with quarterly rebalancing.

    Args:
        prices_wide: Wide price panel (index = date, columns = tickers).
        weights: Mapping ticker -> target weight (must sum to 1).

    Returns:
        Daily portfolio returns.

    Raises:
        KeyError: If a ticker is not in ``prices_wide``.
        ValueError: If the weights are invalid.
    """
    validate_weights(weights)
    missing = [t for t in weights if t not in prices_wide.columns]
    if missing:
        raise KeyError(f"Tickers not in the price panel: {missing}.")
    cols = list(weights)
    rets = to_returns(prices_wide[cols])
    w = np.array([weights[c] for c in cols], dtype=float)
    return quarterly_rebalanced_returns(rets, w)


def equal_weights(tickers: list[str]) -> dict[str, float]:
    """Equal-weight portfolio over ``tickers``.

    Args:
        tickers: Universe.

    Returns:
        Mapping ticker -> ``1 / len(tickers)``.

    Raises:
        ValueError: If ``tickers`` is empty.
    """
    if not tickers:
        raise ValueError("Cannot build an equal-weight portfolio of nothing.")
    return {t: 1.0 / len(tickers) for t in tickers}
