"""Shared fixtures: a small synthetic market driven by sector factors."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.data.dataset import MarketDataset  # noqa: E402

N_SECTORS = 3
STOCKS_PER_SECTOR = 12
TRAIN = {"start": "2019-01-01", "end": "2020-12-31"}
VALIDATION = {"start": "2021-01-01", "end": "2021-12-31"}
FIXTURE_SEED = 7


def make_synthetic_dataset(seed: int = FIXTURE_SEED) -> MarketDataset:
    """Build a rectangular price panel with market + sector factors.

    Args:
        seed: RNG seed.

    Returns:
        A dataset whose benchmark is the (noisy) equal-weight index.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-12-31", "2021-12-31")
    n_days = len(dates)
    market = rng.normal(0.0004, 0.010, n_days)
    tickers, sectors, cols = [], {}, []
    for s in range(N_SECTORS):
        sector_factor = rng.normal(0.0, 0.008, n_days)
        for k in range(STOCKS_PER_SECTOR):
            ticker = f"S{s}K{k:02d}"
            beta = rng.uniform(0.7, 1.3)
            idio = rng.normal(0.0, 0.006, n_days)
            cols.append(beta * market + sector_factor + idio)
            tickers.append(ticker)
            sectors[ticker] = f"Sector{s}"
    returns = np.column_stack(cols)
    returns[0] = 0.0
    prices = pd.DataFrame(100.0 * np.cumprod(1.0 + returns, axis=0),
                          index=dates, columns=tickers)
    bench_ret = returns.mean(axis=1) + rng.normal(0.0, 0.001, n_days)
    bench_ret[0] = 0.0
    benchmark = pd.Series(50.0 * np.cumprod(1.0 + bench_ret), index=dates,
                          name="RSP")
    metadata = {
        "universe": tickers,
        "windows": {"train": TRAIN, "validation": VALIDATION},
        "n_assets": 5,
        "sectors": sectors,
        "target": {"name": "RSP", "column": "rsp_adj"},
    }
    return MarketDataset(metadata=metadata, prices_wide=prices,
                         benchmark=benchmark)


@pytest.fixture(scope="session")
def synthetic_dataset() -> MarketDataset:
    """Session-wide synthetic dataset."""
    return make_synthetic_dataset()
