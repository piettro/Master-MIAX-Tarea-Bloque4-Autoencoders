"""Unit tests for src.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics


@pytest.fixture()
def series() -> tuple[pd.Series, pd.Series]:
    """Two short return series on the same dates."""
    idx = pd.bdate_range("2021-01-01", periods=6)
    a = pd.Series([0.01, -0.02, 0.03, 0.00, 0.01, -0.01], index=idx)
    b = pd.Series([0.00, -0.01, 0.02, 0.01, 0.00, -0.02], index=idx)
    return a, b


def test_tracking_error_matches_definition(series):
    a, b = series
    expected = (a - b).std(ddof=1) * np.sqrt(252)
    assert metrics.annualized_tracking_error(a, b) == pytest.approx(expected)


def test_tracking_error_zero_for_identical(series):
    a, _ = series
    assert metrics.annualized_tracking_error(a, a) == pytest.approx(0.0)


def test_tracking_error_nan_with_one_observation(series):
    a, b = series
    assert np.isnan(metrics.annualized_tracking_error(a[:1], b[:1]))


def test_volatility(series):
    a, _ = series
    assert metrics.annualized_volatility(a) == pytest.approx(
        a.std(ddof=1) * np.sqrt(252)
    )


def test_cagr_constant_growth():
    r = pd.Series([0.001] * 252)
    assert metrics.cagr(r) == pytest.approx(1.001 ** 252 - 1)


def test_cagr_empty_is_nan():
    assert np.isnan(metrics.cagr(pd.Series(dtype=float)))


def test_max_drawdown():
    r = pd.Series([0.10, -0.50, 0.20])
    # Wealth 1.1 -> 0.55 -> 0.66; worst drop from 1.1 to 0.55 = -50 %.
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)


def test_max_drawdown_monotonic_is_zero():
    assert metrics.max_drawdown(pd.Series([0.01, 0.02])) == 0.0


def test_correlation_perfect(series):
    a, _ = series
    assert metrics.correlation(a, 2 * a) == pytest.approx(1.0)


def test_summarize_and_frame(series):
    a, b = series
    summary = metrics.summarize(a, b)
    assert summary.n_days == len(a)
    assert summary.tracking_error == pytest.approx(
        metrics.annualized_tracking_error(a, b)
    )
    frame = summary.to_frame("X")
    assert list(frame.columns) == ["X", "RSP"]
    assert frame.loc["Tracking error", "RSP"] == ""
    assert set(summary.to_dict()) >= {"tracking_error", "correlation"}
