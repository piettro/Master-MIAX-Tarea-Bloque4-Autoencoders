"""Métricas de evaluación del tracker (andamiaje del notebook).

Funciones puras sobre series de retornos diarios. El **tracking error** aquí es
idéntico al que calcula el servidor (core/scoring.py): desviación típica de la
diferencia diaria de retornos entre tracker y benchmark (RSP), anualizada ×√252.

Incluye además el panel de métricas que pide el taller para comparar el tracker
con RSP: TE, correlación, CAGR, volatilidad anualizada y máximo drawdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
#  Retornos
# --------------------------------------------------------------------------- #
def to_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Retornos simples diarios a partir de precios (ajustados)."""
    return prices.pct_change(fill_method=None)


def _quarterly_portfolio_returns(rets: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Retorno diario de la cartera con rebalanceo TRIMESTRAL a los pesos objetivo.

    Se rebalancea a ``weights`` el primer día de cada trimestre natural y se mantiene
    (buy-and-hold: los pesos derivan con los precios) hasta el siguiente. IDÉNTICO al
    servidor (core/scoring.quarterly_portfolio_returns).
    """
    out = pd.Series(index=rets.index, dtype=float)
    quarters = rets.index.to_period("Q")
    for period in pd.unique(quarters):
        mask = np.asarray(quarters == period)
        sub = rets.iloc[mask].fillna(0.0)
        value = (1.0 + sub).cumprod().mul(weights, axis=1).sum(axis=1)  # valor de la cartera
        prev = value.shift(1)
        prev.iloc[0] = 1.0                                             # inicio de trimestre = rebalanceo a los pesos
        out.iloc[mask] = (value / prev - 1.0).to_numpy()
    return out


def portfolio_returns(prices_wide: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Retorno diario del tracker con rebalanceo TRIMESTRAL a los pesos objetivo.

    Igual que RSP (que rebalancea cada trimestre): dentro de cada trimestre natural la
    cartera es buy-and-hold (los pesos derivan) y se reajusta a los pesos objetivo al
    empezar el siguiente. Misma definición que usa el servidor para puntuar.

    ``prices_wide``: precios en formato ancho (index=fecha, columnas=tickers).
    ``weights``: dict ticker -> peso (deben sumar 1).
    """
    cols = list(weights)
    rets = prices_wide[cols].pct_change(fill_method=None)
    w = np.array([weights[c] for c in cols], dtype=float)
    return _quarterly_portfolio_returns(rets, w)


# --------------------------------------------------------------------------- #
#  Métricas
# --------------------------------------------------------------------------- #
def annualized_tracking_error(
    tracker_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """TE anualizado = std(tracker - benchmark) × √periods_per_year."""
    diff = (tracker_returns - benchmark_returns).dropna()
    if len(diff) < 2:
        return float("nan")
    return float(diff.std(ddof=1) * np.sqrt(periods_per_year))


def annualized_volatility(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def cagr(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Tasa de crecimiento anual compuesta a partir de retornos diarios."""
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    total_growth = float((1.0 + r).prod())
    years = len(r) / periods_per_year
    if years <= 0 or total_growth <= 0:
        return float("nan")
    return total_growth ** (1.0 / years) - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Máxima caída desde un máximo previo (valor negativo, p.ej. -0.20)."""
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def correlation(tracker_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    a, b = tracker_returns.align(benchmark_returns, join="inner")
    return float(a.corr(b))


def summary(
    tracker_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict[str, float]:
    """Panel completo de métricas tracker vs benchmark (RSP)."""
    a, b = tracker_returns.align(benchmark_returns, join="inner")
    return {
        "tracking_error": annualized_tracking_error(a, b, periods_per_year),
        "correlation": correlation(a, b),
        "cagr_tracker": cagr(a, periods_per_year),
        "cagr_benchmark": cagr(b, periods_per_year),
        "vol_tracker": annualized_volatility(a, periods_per_year),
        "vol_benchmark": annualized_volatility(b, periods_per_year),
        "maxdd_tracker": max_drawdown(a),
        "maxdd_benchmark": max_drawdown(b),
        "n_days": int(len(a.dropna())),
    }


def summary_frame(
    tracker_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
    tracker_name: str = "Tracker",
) -> pd.DataFrame:
    """Panel legible tracker vs RSP (CAGR, vol, maxDD) + TE y correlación.

    Devuelve un DataFrame con una fila por métrica. Las métricas de relación
    (TE, correlación) solo aplican al par, así que se muestran en la columna del
    tracker y se dejan en blanco para RSP.
    """
    s = summary(tracker_returns, benchmark_returns, periods_per_year)
    rows = {
        "CAGR": (f"{s['cagr_tracker']:.2%}", f"{s['cagr_benchmark']:.2%}"),
        "Vol. anualizada": (f"{s['vol_tracker']:.2%}", f"{s['vol_benchmark']:.2%}"),
        "Máx. drawdown": (f"{s['maxdd_tracker']:.2%}", f"{s['maxdd_benchmark']:.2%}"),
        "Tracking error": (f"{s['tracking_error']:.2%}", ""),
        "Correlación": (f"{s['correlation']:.3f}", ""),
        "Días": (s["n_days"], ""),
    }
    return pd.DataFrame.from_dict(
        {k: {tracker_name: v[0], "RSP": v[1]} for k, v in rows.items()}, orient="index"
    )
