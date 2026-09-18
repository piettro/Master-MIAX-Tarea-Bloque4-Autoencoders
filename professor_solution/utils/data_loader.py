"""Carga de datos del taller vía la API autenticada (andamiaje del notebook).

El alumno NO tiene los ficheros de datos: los descarga de la webapp con su
**token de grupo**. La API solo entrega train+validación (el test nunca sale del
servidor). Ver webapp/ingest.py.

Uso típico en el notebook:

    import data_loader as dl
    data = dl.load_all(API_URL, TOKEN)
    prices = data["prices_wide"]      # precios ancho (fecha × ticker)
    rsp = data["benchmark"]           # serie RSP (fecha)
    meta = data["metadata"]           # universo, ventanas, target
"""

from __future__ import annotations

import io

import pandas as pd
import requests

TIMEOUT = 60


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def get_metadata(api_url: str, token: str) -> dict:
    r = requests.get(f"{api_url}/api/metadata", headers=_headers(token), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _get_parquet(api_url: str, path: str, token: str) -> pd.DataFrame:
    r = requests.get(f"{api_url}{path}", headers=_headers(token), timeout=TIMEOUT)
    if r.status_code == 401:
        raise PermissionError("Token de grupo inválido. Cópialo de tu panel en la webapp.")
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def get_prices(api_url: str, token: str) -> pd.DataFrame:
    """Panel de precios train+val en formato largo: date, symbol, close."""
    return _get_parquet(api_url, "/api/prices", token)


def get_benchmark(api_url: str, token: str) -> pd.DataFrame:
    """Serie RSP train+val en formato largo: date, rsp_adj, rsp_close."""
    return _get_parquet(api_url, "/api/benchmark", token)


def to_wide(prices_long: pd.DataFrame) -> pd.DataFrame:
    """Pasa el panel largo (date, symbol, close) a ancho (index=fecha, cols=ticker)."""
    return prices_long.pivot(index="date", columns="symbol", values="close").sort_index()


def benchmark_series(benchmark_long: pd.DataFrame, column: str = "rsp_adj") -> pd.Series:
    """Serie del benchmark indexada por fecha (por defecto Adj Close de RSP)."""
    s = benchmark_long.set_index("date")[column].sort_index()
    s.name = "RSP"
    return s


def asset_return_matrix(prices_wide: pd.DataFrame, date_from=None, date_to=None):
    """Matriz de retornos para el autoencoder: una fila por ACCIÓN.

    Devuelve ``(X, tickers)`` donde ``X`` tiene forma ``(n_acciones, n_días)`` —
    cada acción es una muestra y sus retornos diarios son las características que
    el encoder proyectará al espacio latente. Se descarta la primera fila (NaN del
    primer retorno) y se restringe a la ventana indicada.
    """
    px = prices_wide
    if date_from is not None:
        px = px[px.index >= pd.Timestamp(date_from)]
    if date_to is not None:
        px = px[px.index <= pd.Timestamp(date_to)]
    rets = px.pct_change(fill_method=None).iloc[1:]
    tickers = list(rets.columns)
    X = rets.T.to_numpy()  # (n_acciones, n_días)
    return X, tickers


def load_all(api_url: str, token: str) -> dict:
    """Descarga metadata + precios + benchmark y prepara estructuras listas."""
    meta = get_metadata(api_url, token)
    prices_long = get_prices(api_url, token)
    bench_long = get_benchmark(api_url, token)
    return {
        "metadata": meta,
        "prices_long": prices_long,
        "prices_wide": to_wide(prices_long),
        "benchmark": benchmark_series(bench_long, meta["target"]["column"]),
        "benchmark_long": bench_long,
    }
