"""Unit tests for src.models (architectures, training, PCA)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models import autoencoder as ae
from src.models.pca_baseline import pca_latent
from src.models.training import encode, train_autoencoder
from src.utils.config import (
    BaselineAEConfig,
    DenoisingAEConfig,
    TrainingConfig,
)
from src.utils.seed import set_global_seed

INPUT_DIM = 40


def test_denoising_architecture_matches_reference():
    model, encoder = ae.build_denoising_autoencoder(INPUT_DIM)
    types = [type(layer).__name__ for layer in model.layers]
    assert types == ["InputLayer", "GaussianNoise", "Dense", "Dense",
                     "Dense", "Dense"]
    assert model.get_layer("input_noise").stddev == pytest.approx(0.3)
    assert model.get_layer("encoder_hidden").units == 64
    latent = model.get_layer(ae.LATENT_LAYER_NAME)
    assert latent.units == 8
    assert latent.get_config()["activation"] == "linear"
    assert tuple(encoder.outputs[0].shape) == (None, 8)
    assert tuple(model.outputs[0].shape) == (None, INPUT_DIM)


def test_denoising_optional_dropout():
    cfg = DenoisingAEConfig(latent_dim=4, hidden_units=8, dropout_rate=0.2)
    model, _ = ae.build_denoising_autoencoder(INPUT_DIM, cfg)
    assert sum(type(layer).__name__ == "Dropout"
               for layer in model.layers) == 2


@pytest.mark.parametrize(
    "cfg",
    [DenoisingAEConfig(latent_dim=0), DenoisingAEConfig(dropout_rate=1.0),
     DenoisingAEConfig(noise_stddev=-0.1)],
)
def test_denoising_invalid_config_raises(cfg):
    with pytest.raises(ValueError):
        ae.build_denoising_autoencoder(INPUT_DIM, cfg)


def test_baseline_architecture():
    model, encoder = ae.build_baseline_autoencoder(
        INPUT_DIM, BaselineAEConfig(latent_dim=10)
    )
    assert model.get_layer(ae.LATENT_LAYER_NAME).get_config()[
        "activation"] == "relu"
    assert tuple(encoder.outputs[0].shape) == (None, 10)


def test_training_and_encoding():
    set_global_seed(42)
    x = np.random.default_rng(0).normal(size=(30, INPUT_DIM))
    model, encoder = ae.build_denoising_autoencoder(
        INPUT_DIM, DenoisingAEConfig(latent_dim=3, hidden_units=8)
    )
    result = train_autoencoder(
        model, x, TrainingConfig(epochs=3, early_stopping_patience=2)
    )
    assert result.epochs_run <= 3
    assert 1 <= result.best_epoch <= result.epochs_run
    assert np.isfinite(result.best_val_loss)
    z = encode(encoder, x)
    assert z.shape == (30, 3)
    # No noise at inference -> deterministic embedding.
    assert np.array_equal(z, encode(encoder, x))


def test_training_rejects_bad_input():
    model, _ = ae.build_denoising_autoencoder(INPUT_DIM)
    with pytest.raises(ValueError):
        train_autoencoder(model, np.full((5, INPUT_DIM), np.nan))
    with pytest.raises(ValueError):
        train_autoencoder(model, np.zeros((5, INPUT_DIM + 1)))


def test_pca_latent():
    x = np.random.default_rng(0).normal(size=(20, 10))
    z = pca_latent(x, n_components=3)
    assert z.shape == (20, 3)
    with pytest.raises(ValueError):
        pca_latent(x, n_components=11)
