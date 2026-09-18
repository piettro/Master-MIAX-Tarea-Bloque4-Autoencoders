"""Unit tests for src.portfolio.backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio import backtest


@pytest.fixture()
def prices() -> pd.DataFrame:
    """Two assets across a quarter boundary."""
    idx = pd.to_datetime(
        ["2021-03-29", "2021-03-30", "2021-03-31", "2021-04-01",
         "2021-04-05"]
    )
    return pd.DataFrame(
        {"A": [100, 110, 121, 121, 133.1], "B": [100, 100, 100, 110, 110]},
        index=idx, dtype=float,
    )


def test_single_asset_equals_asset_returns(prices):
    out = backtest.portfolio_returns(prices, {"A": 1.0})
    expected = prices["A"].pct_change()
    pd.testing.assert_series_equal(
        out.iloc[1:], expected.iloc[1:], check_names=False
    )


def test_weights_drift_within_quarter_and_reset(prices):
    out = backtest.portfolio_returns(prices, {"A": 0.5, "B": 0.5})
    # Day 2 (Q1): A +10 %, B flat -> 5 %.
    assert out.iloc[1] == pytest.approx(0.05)
    # Day 3 (Q1): weights drifted to A=0.55/1.05 -> 0.1 * 0.5238.
    assert out.iloc[2] == pytest.approx(0.1 * 0.55 / 1.05)
    # First day of Q2: rebalanced to 50/50; A flat, B +10 % -> 5 %.
    assert out.iloc[3] == pytest.approx(0.05)


def test_constant_prices_give_zero_returns():
    idx = pd.bdate_range("2021-01-01", periods=5)
    flat = pd.DataFrame({"A": 1.0, "B": 2.0}, index=idx)
    out = backtest.portfolio_returns(flat, {"A": 0.3, "B": 0.7})
    assert np.allclose(out.fillna(0.0), 0.0)


@pytest.mark.parametrize(
    "weights",
    [{}, {"A": -0.5, "B": 1.5}, {"A": 0.5, "B": 0.4}],
)
def test_invalid_weights_raise(prices, weights):
    with pytest.raises(ValueError):
        backtest.portfolio_returns(prices, weights)


def test_unknown_ticker_raises(prices):
    with pytest.raises(KeyError):
        backtest.portfolio_returns(prices, {"ZZZ": 1.0})


def test_equal_weights():
    w = backtest.equal_weights(["A", "B", "C", "D"])
    assert w == {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    with pytest.raises(ValueError):
        backtest.equal_weights([])


def test_mismatched_weight_length_raises(prices):
    with pytest.raises(ValueError):
        backtest.quarterly_rebalanced_returns(
            prices.pct_change(), np.array([1.0])
        )
