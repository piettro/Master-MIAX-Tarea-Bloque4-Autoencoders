"""Autoencoder architectures (Keras).

The competition contract fixes only the input and output size
(``input_dim`` = number of training days); everything in between is free.

Two architectures are provided:

* :func:`build_denoising_autoencoder` — the corrected model (professor's
  reference solution). Input Gaussian noise + one ``tanh`` hidden layer + a
  small **linear** bottleneck + symmetric decoder. With few samples
  (~342 stocks) and many features (~1762 days) a *simple* regularised
  network generalises better than a deep one and beats PCA out of sample.
* :func:`build_baseline_autoencoder` — the original, uncorrected skeleton
  (one ``relu`` layer into a 400-d latent). Kept only for comparison: its
  latent space is barely smaller than needed to copy the input, so it
  memorises (1.4 M parameters) and loses to PCA.
"""

from __future__ import annotations

import logging

from tensorflow import keras
from tensorflow.keras import layers

from src.utils.config import BaselineAEConfig, DenoisingAEConfig

logger = logging.getLogger(__name__)

LATENT_LAYER_NAME = "latent"
OUTPUT_ACTIVATION = "linear"  # z-scored returns can be negative


def _check_dims(input_dim: int, latent_dim: int) -> None:
    """Validate input/latent dimensions.

    Args:
        input_dim: Number of input features.
        latent_dim: Bottleneck size.

    Raises:
        ValueError: If a dimension is not positive.
    """
    if input_dim < 1:
        raise ValueError(f"input_dim must be positive, got {input_dim}.")
    if latent_dim < 1:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
    if latent_dim >= input_dim:
        logger.warning(
            "latent_dim (%d) >= input_dim (%d): the network can learn the "
            "identity instead of compressing.", latent_dim, input_dim,
        )


def build_denoising_autoencoder(
    input_dim: int, config: DenoisingAEConfig | None = None
) -> tuple[keras.Model, keras.Model]:
    """Build the reference denoising autoencoder.

    Architecture::

        input -> GaussianNoise(sigma) -> Dense(h, tanh) [-> Dropout]
              -> Dense(latent_dim, linear)  "latent"
              -> Dense(h, tanh) [-> Dropout] -> Dense(input_dim, linear)

    Args:
        input_dim: Number of input features (training days).
        config: Hyper-parameters (defaults = professor's solution).

    Returns:
        ``(autoencoder, encoder)`` sharing the same weights.

    Raises:
        ValueError: If a dimension or rate is invalid.
    """
    # NOTE: added — required by assignment (section 4) / missing in
    # original submission: denoising regularisation, a hidden non-linear
    # layer and a small linear bottleneck replace the 400-d relu latent.
    cfg = config or DenoisingAEConfig()
    _check_dims(input_dim, cfg.latent_dim)
    if not 0.0 <= cfg.dropout_rate < 1.0:
        raise ValueError(
            f"dropout_rate must be in [0, 1), got {cfg.dropout_rate}."
        )
    if cfg.noise_stddev < 0.0:
        raise ValueError(
            f"noise_stddev must be >= 0, got {cfg.noise_stddev}."
        )

    inputs = keras.Input(shape=(input_dim,), name="returns")
    # Denoising: noise is only active during training. It forces the
    # network to keep the robust common factors and discard idiosyncratic
    # day-to-day noise, acting as a regulariser.
    x = layers.GaussianNoise(cfg.noise_stddev, name="input_noise")(inputs)
    x = layers.Dense(cfg.hidden_units, activation=cfg.hidden_activation,
                     name="encoder_hidden")(x)
    if cfg.dropout_rate > 0.0:
        x = layers.Dropout(cfg.dropout_rate, name="encoder_dropout")(x)
    # Linear bottleneck: signed coordinates (no dead relu units), so the
    # cosine-based selection sees a meaningful direction for every stock.
    latent = layers.Dense(cfg.latent_dim, activation="linear",
                          name=LATENT_LAYER_NAME)(x)
    x = layers.Dense(cfg.hidden_units, activation=cfg.hidden_activation,
                     name="decoder_hidden")(latent)
    if cfg.dropout_rate > 0.0:
        x = layers.Dropout(cfg.dropout_rate, name="decoder_dropout")(x)
    outputs = layers.Dense(input_dim, activation=OUTPUT_ACTIVATION,
                           name="reconstruction")(x)

    autoencoder = keras.Model(inputs, outputs, name="autoencoder")
    encoder = keras.Model(inputs, latent, name="encoder")
    validate_io_shapes(autoencoder, input_dim)
    return autoencoder, encoder


def build_baseline_autoencoder(
    input_dim: int, config: BaselineAEConfig | None = None
) -> tuple[keras.Model, keras.Model]:
    """Build the original (uncorrected) single-layer autoencoder.

    Args:
        input_dim: Number of input features (training days).
        config: Hyper-parameters (defaults = original skeleton).

    Returns:
        ``(autoencoder, encoder)`` sharing the same weights.
    """
    cfg = config or BaselineAEConfig()
    _check_dims(input_dim, cfg.latent_dim)
    inputs = keras.Input(shape=(input_dim,), name="returns")
    latent = layers.Dense(cfg.latent_dim, activation=cfg.latent_activation,
                          name=LATENT_LAYER_NAME)(inputs)
    outputs = layers.Dense(input_dim, activation=OUTPUT_ACTIVATION,
                           name="reconstruction")(latent)
    autoencoder = keras.Model(inputs, outputs, name="autoencoder")
    encoder = keras.Model(inputs, latent, name="encoder")
    validate_io_shapes(autoencoder, input_dim)
    return autoencoder, encoder


def validate_io_shapes(model: keras.Model, input_dim: int) -> None:
    """Enforce the contract: input and output must both be ``input_dim``.

    Args:
        model: The autoencoder.
        input_dim: Expected feature count.

    Raises:
        ValueError: If input or output shape differs from the contract.
    """
    expected = (None, input_dim)
    in_shape = tuple(model.inputs[0].shape)
    out_shape = tuple(model.outputs[0].shape)
    if in_shape != expected:
        raise ValueError(f"Input shape {in_shape} != {expected}.")
    if out_shape != expected:
        raise ValueError(
            f"Output shape {out_shape} != {expected}: the decoder must "
            "reconstruct input_dim features."
        )
