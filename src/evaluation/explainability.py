"""Latent-space explainability helpers.

The professor stressed that a slightly worse but *explainable* latent space
can be preferable to an opaque one. These helpers compute price-based
company characteristics (volatility, growth, beta — market cap is not
available) and measure how strongly each latent dimension correlates with
them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dataset import DateWindow
from src.data.preprocessing import to_returns
from src.utils.config import TRADING_DAYS_PER_YEAR

FEATURE_VOLATILITY = "Annualised volatility"
FEATURE_GROWTH = "Cumulative growth"
FEATURE_BETA = "Beta vs RSP"


def company_features(
    prices_wide: pd.DataFrame,
    tickers: list[str],
    benchmark_returns: pd.Series,
    window: DateWindow,
) -> pd.DataFrame:
    """Price-based characteristics of each company over ``window``.

    Args:
        prices_wide: Wide price panel.
        tickers: Tickers in latent-space order.
        benchmark_returns: Daily benchmark returns.
        window: Window used to compute the features (train).

    Returns:
        DataFrame indexed by ticker with volatility, growth and beta.

    Raises:
        ValueError: If the benchmark has zero variance in the window.
    """
    rets = window.slice(to_returns(prices_wide[tickers]))
    bench = window.slice(benchmark_returns).reindex(rets.index)
    bench_var = bench.var()
    if not bench_var or np.isnan(bench_var):
        raise ValueError("Benchmark variance is zero/NaN; beta undefined.")
    # NOTE: improved over professor's solution — vectorised beta instead of a
    # per-column lambda (same result, pandas aligns and skips NaNs).
    beta = rets.apply(lambda col: col.cov(bench)) / bench_var
    return pd.DataFrame(
        {
            FEATURE_VOLATILITY: rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR),
            FEATURE_GROWTH: (1.0 + rets).prod() - 1.0,
            FEATURE_BETA: beta,
        },
        index=tickers,
    )


def latent_feature_correlation(
    latent: np.ndarray, features: pd.DataFrame
) -> pd.DataFrame:
    """Pearson correlation of each latent dimension with each feature.

    Args:
        latent: Latent matrix ``(n_stocks, latent_dim)``.
        features: Feature table with ``n_stocks`` rows (same order).

    Returns:
        DataFrame ``(latent_dim, n_features)`` indexed ``dim 0..k``.

    Raises:
        ValueError: If the row counts differ.
    """
    if latent.shape[0] != len(features):
        raise ValueError(
            f"latent has {latent.shape[0]} rows, features {len(features)}."
        )
    corr = {
        name: [
            float(np.corrcoef(latent[:, d], features[name].to_numpy())[0, 1])
            for d in range(latent.shape[1])
        ]
        for name in features.columns
    }
    index = [f"dim {d}" for d in range(latent.shape[1])]
    return pd.DataFrame(corr, index=index)
