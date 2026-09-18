"""Training and encoding utilities for the autoencoders."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from tensorflow import keras

from src.utils.config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a training run.

    Attributes:
        history: Per-epoch ``loss`` / ``val_loss`` values.
        epochs_run: Number of epochs actually executed.
        best_epoch: 1-based epoch with the lowest ``val_loss`` (its weights
            are restored).
        best_val_loss: Lowest validation loss.
        stopped_early: Whether early stopping triggered.
    """

    history: dict[str, list[float]]
    epochs_run: int
    best_epoch: int
    best_val_loss: float
    stopped_early: bool


def train_autoencoder(
    autoencoder: keras.Model,
    x_train: np.ndarray,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Compile and fit the autoencoder to reconstruct ``x_train``.

    Hyper-parameters are the ones fixed by the workshop scaffold (Adam
    1e-3, MSE, 200 epochs, batch 32, 10 % hold-out rows, early stopping
    with patience 15 and best-weight restoration).

    Args:
        autoencoder: Uncompiled Keras model.
        x_train: Scaled training matrix ``(n_stocks, n_days)``.
        config: Training hyper-parameters.

    Returns:
        The training summary.

    Raises:
        ValueError: If ``x_train`` is not 2-D, contains non-finite values
            or does not match the model input.
    """
    cfg = config or TrainingConfig()
    if x_train.ndim != 2:
        raise ValueError(f"x_train must be 2-D, got ndim={x_train.ndim}.")
    if not np.isfinite(x_train).all():
        raise ValueError("x_train contains NaN or infinite values.")
    expected_dim = autoencoder.inputs[0].shape[-1]
    if x_train.shape[1] != expected_dim:
        raise ValueError(
            f"x_train has {x_train.shape[1]} features, model expects "
            f"{expected_dim}."
        )

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate), loss=cfg.loss
    )
    early_stopping = keras.callbacks.EarlyStopping(
        patience=cfg.early_stopping_patience, restore_best_weights=True
    )
    logger.info(
        "Training %s (%s params) on %s", autoencoder.name,
        f"{autoencoder.count_params():,}", x_train.shape,
    )
    fit = autoencoder.fit(
        x_train, x_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=cfg.validation_split,
        callbacks=[early_stopping],
        verbose=0,
    )
    history = {k: [float(v) for v in vals]
               for k, vals in fit.history.items()}
    val_loss = history["val_loss"]
    best_epoch = int(np.argmin(val_loss)) + 1
    result = TrainingResult(
        history=history,
        epochs_run=len(history["loss"]),
        best_epoch=best_epoch,
        best_val_loss=float(min(val_loss)),
        # stopped_epoch is 0 when early stopping did not trigger.
        stopped_early=bool(early_stopping.stopped_epoch),
    )
    logger.info(
        "Epochs run: %d/%d | early stop: %s | best val_loss %.6f at "
        "epoch %d (weights restored)", result.epochs_run, cfg.epochs,
        result.stopped_early, result.best_val_loss, result.best_epoch,
    )
    return result


def encode(encoder: keras.Model, x: np.ndarray) -> np.ndarray:
    """Project every stock into the latent space.

    Noise/dropout layers are inactive at inference, so the embedding is
    deterministic.

    Args:
        encoder: Trained encoder.
        x: Scaled matrix ``(n_stocks, n_days)``.

    Returns:
        Latent coordinates ``(n_stocks, latent_dim)`` as float64.
    """
    return np.asarray(encoder.predict(x, verbose=0), dtype=np.float64)
