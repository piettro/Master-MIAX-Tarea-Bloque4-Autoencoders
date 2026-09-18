"""Evaluation of a tracker (weights) against the benchmark per window."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.dataset import DateWindow
from src.evaluation.metrics import (
    PerformanceSummary,
    annualized_tracking_error,
    summarize,
)
from src.portfolio.backtest import portfolio_returns


@dataclass(frozen=True)
class TrackerEvaluation:
    """Tracker composition and its out-of-sample quality.

    Attributes:
        name: Label (e.g. ``"Autoencoder"``, ``"PCA"``).
        weights: Mapping ticker -> weight.
        returns: Daily tracker returns over the whole panel.
        te_train: Annualised TE on the training window.
        te_validation: Annualised TE on the validation window.
        validation_summary: Full metric panel on validation.
    """

    name: str
    weights: dict[str, float]
    returns: pd.Series
    te_train: float
    te_validation: float
    validation_summary: PerformanceSummary


def evaluate_tracker(
    name: str,
    weights: dict[str, float],
    prices_wide: pd.DataFrame,
    benchmark_returns: pd.Series,
    train_window: DateWindow,
    validation_window: DateWindow,
) -> TrackerEvaluation:
    """Backtest a tracker and measure it on train and validation.

    Args:
        name: Label of the tracker.
        weights: Mapping ticker -> target weight.
        prices_wide: Wide price panel.
        benchmark_returns: Daily benchmark returns.
        train_window: Training window.
        validation_window: Validation window.

    Returns:
        The evaluation.
    """
    returns = portfolio_returns(prices_wide, weights)
    te_train = annualized_tracking_error(
        train_window.slice(returns), train_window.slice(benchmark_returns)
    )
    val_summary = summarize(
        validation_window.slice(returns),
        validation_window.slice(benchmark_returns),
    )
    return TrackerEvaluation(
        name=name,
        weights=dict(weights),
        returns=returns,
        te_train=te_train,
        te_validation=val_summary.tracking_error,
        validation_summary=val_summary,
    )
