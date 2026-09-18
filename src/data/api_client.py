"""Authenticated download of the workshop data, with a local parquet cache.

The API only serves train + validation data; the test window never leaves
the server. The first successful download is cached under ``data/raw/`` so
subsequent runs are fast, work offline and are byte-for-byte reproducible.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.data.dataset import MarketDataset
from src.data.preprocessing import benchmark_series, to_wide
from src.utils.config import (
    BENCHMARK_FILENAME,
    DATA_DIR,
    DEFAULT_BENCHMARK_COLUMN,
    ENDPOINT_BENCHMARK,
    ENDPOINT_METADATA,
    ENDPOINT_PRICES,
    HTTP_TIMEOUT_GET_S,
    METADATA_FILENAME,
    PRICES_FILENAME,
    TOKEN_ENV_VAR,
)

logger = logging.getLogger(__name__)


def auth_headers(token: str) -> dict[str, str]:
    """Build the bearer-token header expected by the API.

    Args:
        token: Group token.

    Returns:
        HTTP headers.

    Raises:
        ValueError: If the token is empty.
    """
    if not token:
        raise ValueError("The group token is empty.")
    return {"Authorization": f"Bearer {token}"}


def _get(api_url: str, path: str, token: str) -> requests.Response:
    """Perform an authenticated GET and translate common failures.

    Args:
        api_url: API base URL.
        path: Endpoint path (starting with ``/``).
        token: Group token.

    Returns:
        The successful response.

    Raises:
        PermissionError: If the token is rejected (HTTP 401/403).
        ConnectionError: On network errors or other HTTP errors.
    """
    url = f"{api_url}{path}"
    try:
        response = requests.get(
            url, headers=auth_headers(token), timeout=HTTP_TIMEOUT_GET_S
        )
    except requests.RequestException as exc:
        raise ConnectionError(f"Could not reach {url}: {exc}") from exc
    if response.status_code in (401, 403):
        raise PermissionError(
            "Invalid group token. Copy it from your group panel in the "
            f"web-app and export it as {TOKEN_ENV_VAR}."
        )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise ConnectionError(f"GET {url} failed: {exc}") from exc
    return response


def fetch_metadata(api_url: str, token: str) -> dict[str, Any]:
    """Download the competition metadata (universe, windows, target...).

    Args:
        api_url: API base URL.
        token: Group token.

    Returns:
        Parsed metadata.
    """
    return _get(api_url, ENDPOINT_METADATA, token).json()


def fetch_parquet(api_url: str, path: str, token: str) -> pd.DataFrame:
    """Download a parquet table from the API.

    Args:
        api_url: API base URL.
        path: Endpoint path.
        token: Group token.

    Returns:
        The decoded table.
    """
    content = _get(api_url, path, token).content
    return pd.read_parquet(io.BytesIO(content))


def _cache_paths(cache_dir: Path) -> tuple[Path, Path, Path]:
    """Return the metadata, prices and benchmark cache file paths."""
    return (
        cache_dir / METADATA_FILENAME,
        cache_dir / PRICES_FILENAME,
        cache_dir / BENCHMARK_FILENAME,
    )


def cache_exists(cache_dir: Path = DATA_DIR) -> bool:
    """Tell whether a complete local cache is available.

    Args:
        cache_dir: Cache directory.

    Returns:
        ``True`` when all three cache files exist.
    """
    return all(p.is_file() for p in _cache_paths(cache_dir))


def download_to_cache(
    api_url: str, token: str, cache_dir: Path = DATA_DIR
) -> None:
    """Download metadata, prices and benchmark and store them locally.

    Args:
        api_url: API base URL.
        token: Group token.
        cache_dir: Destination directory (created if needed).

    Raises:
        OSError: If the cache cannot be written.
    """
    meta_path, prices_path, bench_path = _cache_paths(cache_dir)
    logger.info("Downloading data from %s", api_url)
    metadata = fetch_metadata(api_url, token)
    prices_long = fetch_parquet(api_url, ENDPOINT_PRICES, token)
    bench_long = fetch_parquet(api_url, ENDPOINT_BENCHMARK, token)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        prices_long.to_parquet(prices_path, index=False)
        bench_long.to_parquet(bench_path, index=False)
    except OSError as exc:
        raise OSError(f"Could not write cache to {cache_dir}: {exc}") from exc
    logger.info(
        "Cached %d price rows and %d benchmark rows in %s",
        len(prices_long), len(bench_long), cache_dir,
    )


def load_from_cache(cache_dir: Path = DATA_DIR) -> MarketDataset:
    """Build a :class:`MarketDataset` from the local cache.

    Args:
        cache_dir: Cache directory.

    Returns:
        The dataset.

    Raises:
        FileNotFoundError: If any cache file is missing.
        ValueError: If a cache file is corrupt.
    """
    meta_path, prices_path, bench_path = _cache_paths(cache_dir)
    missing = [p.name for p in (meta_path, prices_path, bench_path)
               if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing cached data files {missing} in {cache_dir}."
        )
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        prices_long = pd.read_parquet(prices_path)
        bench_long = pd.read_parquet(bench_path)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Corrupt cache in {cache_dir} ({exc}). Delete the folder or "
            "run with --refresh-data."
        ) from exc
    column = metadata.get("target", {}).get(
        "column", DEFAULT_BENCHMARK_COLUMN
    )
    return MarketDataset(
        metadata=metadata,
        prices_wide=to_wide(prices_long),
        benchmark=benchmark_series(bench_long, column),
    )


def load_dataset(
    api_url: str,
    token: str | None,
    cache_dir: Path = DATA_DIR,
    refresh: bool = False,
) -> MarketDataset:
    """Return the dataset, downloading it only when needed.

    Args:
        api_url: API base URL.
        token: Group token (only required when the cache is missing or a
            refresh is requested).
        cache_dir: Cache directory.
        refresh: Force a new download even if the cache exists.

    Returns:
        The dataset.

    Raises:
        FileNotFoundError: If there is no cache and no token to download
            the data with.
    """
    if refresh or not cache_exists(cache_dir):
        if not token:
            raise FileNotFoundError(
                f"No cached data in {cache_dir} and no group token set.\n"
                f"  1. Copy your token from the web-app group panel.\n"
                f"  2. export {TOKEN_ENV_VAR}=<token>   (or put "
                f"{TOKEN_ENV_VAR}=<token> in a .env file at the project "
                "root)\n"
                "  3. Run python main.py again; the data will be cached."
            )
        download_to_cache(api_url, token, cache_dir)
    else:
        logger.info("Loading cached data from %s", cache_dir)
    return load_from_cache(cache_dir)
