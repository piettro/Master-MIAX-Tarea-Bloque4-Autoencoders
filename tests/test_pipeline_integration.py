"""End-to-end integration test on the synthetic market (no network)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.evaluation.explainability import (
    company_features,
    latent_feature_correlation,
)
from src.data.preprocessing import to_returns
from src.pipeline import run_pipeline, save_figures, save_results
from src.utils.config import (
    DenoisingAEConfig,
    RunConfig,
    TrainingConfig,
)

FAST_TRAINING = TrainingConfig(epochs=4, early_stopping_patience=2)


@pytest.fixture(scope="module")
def result(synthetic_dataset, tmp_path_factory):
    config = RunConfig(
        denoising=DenoisingAEConfig(latent_dim=3, hidden_units=16),
        training=FAST_TRAINING,
        results_dir=tmp_path_factory.mktemp("results"),
    )
    return run_pipeline(synthetic_dataset, config)


def test_tracker_is_valid(result, synthetic_dataset):
    weights = result.autoencoder_eval.weights
    assert len(weights) == synthetic_dataset.n_assets
    assert sum(weights.values()) == pytest.approx(1.0)
    assert set(weights) <= set(result.tickers)
    assert result.latent.shape == (len(result.tickers), 3)


def test_metrics_are_finite(result):
    for ev in (result.autoencoder_eval, result.pca_eval):
        assert np.isfinite(ev.te_train)
        assert np.isfinite(ev.te_validation)
        assert ev.te_validation >= 0
    assert np.isfinite(result.ew_floor.tracking_error)
    assert result.comparison_frame().shape == (2, 2)


def test_run_is_reproducible(synthetic_dataset, result):
    again = run_pipeline(synthetic_dataset, result.config)
    np.testing.assert_allclose(again.latent, result.latent)
    assert again.autoencoder_eval.weights == result.autoencoder_eval.weights


def test_outputs_are_written(result, synthetic_dataset, tmp_path):
    metrics_path = save_results(result, tmp_path)
    metrics = json.loads(metrics_path.read_text())
    assert metrics["seed"] == 42
    assert {"autoencoder", "pca", "beats_pca_validation"} <= set(metrics)
    assert (tmp_path / "tracker.csv").is_file()
    assert (tmp_path / "latent.csv").is_file()
    figures = save_figures(result, synthetic_dataset, tmp_path / "figs")
    assert all(p.is_file() for p in figures)
    assert len(figures) == 6


def test_explainability_shapes(result, synthetic_dataset):
    ds = synthetic_dataset
    feats = company_features(ds.prices_wide, result.tickers,
                             to_returns(ds.benchmark), ds.train_window)
    assert feats.shape == (len(result.tickers), 3)
    corr = latent_feature_correlation(result.latent, feats)
    assert corr.shape == (3, 3)
    assert (corr.abs().to_numpy() <= 1.0 + 1e-9).all()
