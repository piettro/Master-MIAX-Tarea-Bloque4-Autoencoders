"""Unit tests for src.data (preprocessing, dataset, API client)."""

from __future__ import annotations

import io
import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.data import api_client, preprocessing
from src.data.dataset import DateWindow


def test_to_wide_pivots_and_sorts():
    long = pd.DataFrame({
        "date": ["2021-01-05", "2021-01-04", "2021-01-04", "2021-01-05"],
        "symbol": ["A", "A", "B", "B"],
        "close": [2.0, 1.0, 3.0, 4.0],
    })
    wide = preprocessing.to_wide(long)
    assert list(wide.columns) == ["A", "B"]
    assert wide.index.is_monotonic_increasing
    assert wide.loc["2021-01-05", "B"] == 4.0


def test_to_wide_missing_column():
    with pytest.raises(ValueError):
        preprocessing.to_wide(pd.DataFrame({"date": [], "close": []}))


def test_benchmark_series():
    long = pd.DataFrame({"date": ["2021-01-05", "2021-01-04"],
                         "rsp_adj": [2.0, 1.0], "rsp_close": [9.0, 9.0]})
    s = preprocessing.benchmark_series(long, "rsp_adj")
    assert s.name == "RSP"
    assert s.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError):
        preprocessing.benchmark_series(long, "nope")


def test_date_window_slice_and_validation():
    idx = pd.bdate_range("2021-01-01", "2021-01-31")
    s = pd.Series(range(len(idx)), index=idx)
    w = DateWindow("w", pd.Timestamp("2021-01-10"),
                   pd.Timestamp("2021-01-15"))
    assert w.slice(s).index.min() >= w.start
    assert w.slice(s).index.max() <= w.end
    with pytest.raises(ValueError):
        DateWindow("bad", pd.Timestamp("2021-02-01"),
                   pd.Timestamp("2021-01-01"))
    with pytest.raises(ValueError):
        DateWindow.from_metadata("x", {"start": "2021-01-01"})


def test_asset_return_matrix_shape(synthetic_dataset):
    ds = synthetic_dataset
    x, tickers = preprocessing.asset_return_matrix(ds.prices_wide,
                                                   ds.train_window)
    n_prices = len(ds.train_window.slice(ds.prices_wide))
    assert x.shape == (ds.prices_wide.shape[1], n_prices - 1)
    assert tickers == list(ds.prices_wide.columns)
    assert np.isfinite(x).all()


def test_asset_return_matrix_rejects_missing(synthetic_dataset):
    prices = synthetic_dataset.prices_wide.copy()
    prices.iloc[10, 0] = np.nan
    with pytest.raises(ValueError, match="Missing returns"):
        preprocessing.asset_return_matrix(prices,
                                          synthetic_dataset.train_window)


def test_scaler_standardises_each_day():
    rng = np.random.default_rng(0)
    x = rng.normal(0.01, 0.05, (30, 8))
    _, scaled = preprocessing.fit_scaler(x)
    # Each column (day) is a cross-sectional z-score.
    assert np.allclose(scaled.mean(axis=0), 0.0)
    assert np.allclose(scaled.std(axis=0), 1.0)


def _fake_response(status: int, *, json_body=None, content=b""):
    resp = mock.Mock()
    resp.status_code = status
    resp.json.return_value = json_body
    resp.content = content
    resp.raise_for_status.return_value = None
    return resp


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def test_download_and_cache_roundtrip(tmp_path):
    meta = {"windows": {}, "target": {"column": "rsp_adj"}}
    prices = pd.DataFrame({"date": pd.to_datetime(["2021-01-04"] * 2),
                           "symbol": ["A", "B"], "close": [1.0, 2.0]})
    bench = pd.DataFrame({"date": pd.to_datetime(["2021-01-04"]),
                          "rsp_adj": [3.0], "rsp_close": [3.0]})
    responses = {
        "/api/metadata": _fake_response(200, json_body=meta),
        "/api/prices": _fake_response(200, content=_parquet_bytes(prices)),
        "/api/benchmark": _fake_response(200,
                                         content=_parquet_bytes(bench)),
    }

    def fake_get(url, headers, timeout):
        assert headers == {"Authorization": "Bearer tok"}
        return responses[url.replace("http://x", "")]

    with mock.patch.object(api_client.requests, "get", side_effect=fake_get):
        ds = api_client.load_dataset("http://x", "tok", tmp_path)
    assert api_client.cache_exists(tmp_path)
    assert json.loads((tmp_path / "metadata.json").read_text()) == meta
    assert list(ds.prices_wide.columns) == ["A", "B"]
    # Second call uses the cache: no token needed, no HTTP call.
    with mock.patch.object(api_client.requests, "get") as get:
        again = api_client.load_dataset("http://x", None, tmp_path)
        get.assert_not_called()
    assert again.benchmark.iloc[0] == 3.0


def test_missing_cache_and_token_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="MIAX_AE_TOKEN"):
        api_client.load_dataset("http://x", None, tmp_path)


def test_invalid_token_raises_permission_error():
    with mock.patch.object(api_client.requests, "get",
                           return_value=_fake_response(401)):
        with pytest.raises(PermissionError):
            api_client.fetch_metadata("http://x", "bad")


def test_network_error_is_wrapped():
    err = api_client.requests.ConnectionError("down")
    with mock.patch.object(api_client.requests, "get", side_effect=err):
        with pytest.raises(ConnectionError, match="Could not reach"):
            api_client.fetch_metadata("http://x", "tok")
