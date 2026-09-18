"""Unit tests for the fixed selection contract (src.portfolio.selection)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.portfolio import selection


@pytest.fixture()
def latent() -> tuple[np.ndarray, list[str]]:
    """Four well-separated directions with 5 stocks each."""
    rng = np.random.default_rng(0)
    centers = np.eye(4) * 5.0
    x = np.vstack([c + rng.normal(0, 0.1, (5, 4)) for c in centers])
    tickers = [f"T{i:02d}" for i in range(len(x))]
    return x, tickers


def test_output_is_sorted_and_weights_sum_to_one(latent):
    x, tickers = latent
    sel, w = selection.select_tracker(x, tickers, n_assets=4)
    assert sel == sorted(sel)
    assert len(sel) == len(set(sel)) == 4
    assert sum(w) == pytest.approx(1.0)
    # Four clusters of five stocks -> equal weights.
    assert w == pytest.approx([0.25] * 4)


def test_one_representative_per_group(latent):
    x, tickers = latent
    sel, _ = selection.select_tracker(x, tickers, n_assets=4)
    groups = {int(t[1:]) // 5 for t in sel}
    assert groups == {0, 1, 2, 3}


def test_deterministic(latent):
    x, tickers = latent
    assert selection.select_tracker(x, tickers, 4) == \
        selection.select_tracker(x, tickers, 4)


def test_scale_invariance_cosine(latent):
    """Cosine metric: rescaling a row must not change the selection."""
    x, tickers = latent
    scaled = x.copy()
    scaled[3] *= 10.0
    assert selection.select_tracker(x, tickers, 4) == \
        selection.select_tracker(scaled, tickers, 4)


@pytest.mark.parametrize(
    "coords, tickers, n_assets",
    [
        (np.zeros(5), ["a"] * 5, 1),                 # not 2-D
        (np.ones((3, 2)), ["a", "b"], 1),            # size mismatch
        (np.ones((2, 2)), ["a", "a"], 1),            # duplicates
        (np.eye(3), ["a", "b", "c"], 4),             # too many assets
    ],
)
def test_invalid_inputs_raise(coords, tickers, n_assets):
    with pytest.raises(ValueError):
        selection.select_tracker(coords, tickers, n_assets=n_assets)


def test_signature_matches_contract():
    payload = ("v=1.1.1|seed=42|n_init=10|algo=lloyd|metric=cosine|"
               "n_assets=25")
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    assert selection.selection_signature(25) == f"1.1.1:{digest}"
