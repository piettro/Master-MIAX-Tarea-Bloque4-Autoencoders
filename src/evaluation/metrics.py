"""Tracker-vs-benchmark performance metrics.

All functions operate on daily simple-return series. The tracking error is
identical to the server's (``core/scoring.py``): sample standard deviation
(ddof=1) of the daily return difference, annualised by ``sqrt(252)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.utils.config import BENCHMARK_NAME, TRADING_DAYS_PER_YEAR

MIN_OBSERVATIONS = 2


def annualized_tracking_error(
    tracker_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised tracking error: ``std(tracker - bench) * sqrt(P)``.

    Args:
        tracker_returns: Daily tracker returns.
        benchmark_returns: Daily benchmark returns.
        periods_per_year: Annualisation factor.

    Returns:
        The tracking error, or NaN with fewer than two observations.
    """
    diff = (tracker_returns - benchmark_returns).dropna()
    if len(diff) < MIN_OBSERVATIONS:
        return float("nan")
    return float(diff.std(ddof=1) * np.sqrt(periods_per_year))


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Annualised volatility ``std(r) * sqrt(P)``.

    Args:
        returns: Daily returns.
        periods_per_year: Annualisation factor.

    Returns:
        The volatility, or NaN with fewer than two observations.
    """
    r = returns.dropna()
    if len(r) < MIN_OBSERVATIONS:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def cagr(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Compound annual growth rate from daily returns.

    Args:
        returns: Daily returns.
        periods_per_year: Annualisation factor.

    Returns:
        ``prod(1 + r) ** (P / n) - 1``, or NaN if undefined.
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")
    total_growth = float((1.0 + r).prod())
    years = len(r) / periods_per_year
    if total_growth <= 0:
        return float("nan")
    return total_growth ** (1.0 / years) - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough loss of the wealth curve (negative number).

    Args:
        returns: Daily returns.

    Returns:
        The maximum drawdown (e.g. ``-0.20``), or NaN if empty.
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def correlation(
    tracker_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """Pearson correlation of daily returns on the common dates.

    Args:
        tracker_returns: Daily tracker returns.
        benchmark_returns: Daily benchmark returns.

    Returns:
        The correlation coefficient.
    """
    a, b = tracker_returns.align(benchmark_returns, join="inner")
    return float(a.corr(b))


@dataclass(frozen=True)
class PerformanceSummary:
    """Full metric panel of a tracker against the benchmark.

    Attributes:
        tracking_error: Annualised TE (competition score).
        correlation: Correlation of daily returns.
        cagr_tracker: Tracker CAGR.
        cagr_benchmark: Benchmark CAGR.
        vol_tracker: Tracker annualised volatility.
        vol_benchmark: Benchmark annualised volatility.
        maxdd_tracker: Tracker maximum drawdown.
        maxdd_benchmark: Benchmark maximum drawdown.
        n_days: Number of common observations.
    """

    tracking_error: float
    correlation: float
    cagr_tracker: float
    cagr_benchmark: float
    vol_tracker: float
    vol_benchmark: float
    maxdd_tracker: float
    maxdd_benchmark: float
    n_days: int

    def to_dict(self) -> dict[str, float]:
        """Return the metrics as a plain dictionary."""
        return asdict(self)

    def to_frame(self, tracker_name: str = "Tracker") -> pd.DataFrame:
        """Readable side-by-side table (tracker vs benchmark).

        Relational metrics (TE, correlation) only apply to the pair, so
        they are shown in the tracker column and left blank for RSP.

        Args:
            tracker_name: Column label for the tracker.

        Returns:
            One row per metric, formatted as strings.
        """
        rows = {
            "CAGR": (f"{self.cagr_tracker:.2%}",
                     f"{self.cagr_benchmark:.2%}"),
            "Annualised vol.": (f"{self.vol_tracker:.2%}",
                                f"{self.vol_benchmark:.2%}"),
            "Max drawdown": (f"{self.maxdd_tracker:.2%}",
                             f"{self.maxdd_benchmark:.2%}"),
            "Tracking error": (f"{self.tracking_error:.2%}", ""),
            "Correlation": (f"{self.correlation:.3f}", ""),
            "Days": (str(self.n_days), ""),
        }
        return pd.DataFrame.from_dict(
            {k: {tracker_name: v[0], BENCHMARK_NAME: v[1]}
             for k, v in rows.items()},
            orient="index",
        )


def summarize(
    tracker_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PerformanceSummary:
    """Compute the full metric panel on the common dates.

    Args:
        tracker_returns: Daily tracker returns.
        benchmark_returns: Daily benchmark returns.
        periods_per_year: Annualisation factor.

    Returns:
        The performance summary.
    """
    a, b = tracker_returns.align(benchmark_returns, join="inner")
    return PerformanceSummary(
        tracking_error=annualized_tracking_error(a, b, periods_per_year),
        correlation=correlation(a, b),
        cagr_tracker=cagr(a, periods_per_year),
        cagr_benchmark=cagr(b, periods_per_year),
        vol_tracker=annualized_volatility(a, periods_per_year),
        vol_benchmark=annualized_volatility(b, periods_per_year),
        maxdd_tracker=max_drawdown(a),
        maxdd_benchmark=max_drawdown(b),
        n_days=int(len(a.dropna())),
    )
