"""Submission client for the competition web-app.

Builds the JSON contract (latent space + tracker + model metadata) and posts
it with the group token.

* A normal submission is evaluated on **validation** (live leaderboard) and
  can be repeated — but resubmitting while staring at validation is
  overfitting to validation.
* ``is_final=True`` triggers the **irreversible** single test evaluation
  and freezes the model; it additionally requires ``confirm_final=True``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import requests
from tensorflow import keras

from src.utils.config import ENDPOINT_SUBMISSIONS, HTTP_TIMEOUT_POST_S, SEED

logger = logging.getLogger(__name__)

# Official seed: the server rejects submissions trained with any other.
REQUIRED_SEED = SEED
# Layer config keys worth showing on the leaderboard.
ARCHITECTURE_KEYS = ("units", "activation", "rate", "stddev")


def architecture_summary(model: keras.Model) -> dict[str, Any]:
    """Summarise a Keras model's layers for the leaderboard.

    Args:
        model: The trained autoencoder.

    Returns:
        ``{"layers": [...], "n_params": int | None}``.
    """
    layers_info = []
    for layer in model.layers:
        cfg = layer.get_config()
        info: dict[str, Any] = {"name": layer.name,
                                "type": type(layer).__name__}
        info.update({k: cfg[k] for k in ARCHITECTURE_KEYS
                     if cfg.get(k) is not None})
        layers_info.append(info)
    try:
        n_params: int | None = int(model.count_params())
    except ValueError:  # model not built
        n_params = None
    return {"layers": layers_info, "n_params": n_params}


def build_payload(
    latent_tickers: Sequence[str],
    latent_coordinates: np.ndarray,
    tracker_tickers: Sequence[str],
    tracker_weights: Sequence[float],
    competition_id: str | None = None,
    model_metadata: dict[str, Any] | None = None,
    is_final: bool = False,
) -> dict[str, Any]:
    """Build the submission JSON (contract spec. 4.3).

    Args:
        latent_tickers: Universe tickers in latent order.
        latent_coordinates: Latent matrix ``(N, latent_dim)``.
        tracker_tickers: Selected tickers.
        tracker_weights: Their weights.
        competition_id: Optional competition identifier.
        model_metadata: Seed, latent size, architecture, etc.
        is_final: Whether this is the irreversible test submission.

    Returns:
        The JSON-serialisable payload.

    Raises:
        ValueError: If lengths are inconsistent.
    """
    coords = np.asarray(latent_coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[0] != len(latent_tickers):
        raise ValueError(
            f"latent_coordinates shape {coords.shape} does not match "
            f"{len(latent_tickers)} tickers."
        )
    if len(tracker_tickers) != len(tracker_weights):
        raise ValueError("tracker_tickers and tracker_weights differ in "
                         "length.")
    return {
        "competition_id": competition_id,
        "model_metadata": model_metadata or {},
        "latent_space": {
            "tickers": list(latent_tickers),
            "coordinates": coords.tolist(),
        },
        "tracker": {
            "tickers": list(tracker_tickers),
            "weights": np.asarray(tracker_weights, dtype=float).tolist(),
        },
        "request_test_evaluation": False,
        "is_final_submission": bool(is_final),
    }


def _check_seed(model_metadata: dict[str, Any] | None) -> None:
    """Ensure the metadata carries the official seed.

    Args:
        model_metadata: Submission metadata.

    Raises:
        ValueError: If the seed is missing or not the official one.
    """
    seed = (model_metadata or {}).get("seed")
    try:
        seed_ok = seed is not None and int(seed) == REQUIRED_SEED
    except (TypeError, ValueError):
        seed_ok = False
    if not seed_ok:
        raise ValueError(
            f"SEED must be {REQUIRED_SEED}: the competition is reproducible "
            f"and everyone trains with the same seed (got {seed!r}). "
            "Restore it, retrain and resubmit (the server rejects it too)."
        )


def submit(
    api_url: str,
    token: str,
    payload: dict[str, Any],
    confirm_final: bool = False,
) -> dict[str, Any]:
    """Send a submission and return the server response (includes the TE).

    Args:
        api_url: API base URL.
        token: Group token.
        payload: Output of :func:`build_payload`.
        confirm_final: Must be ``True`` for a final (test) submission.

    Returns:
        The server's JSON response.

    Raises:
        ValueError: On a final submission without confirmation or with a
            wrong seed.
        ConnectionError: On network errors.
        RuntimeError: If the server rejects the submission.
    """
    if payload.get("is_final_submission") and not confirm_final:
        raise ValueError(
            "FINAL evaluation is irreversible: it freezes the model. "
            "Call again with confirm_final=True if you are sure."
        )
    _check_seed(payload.get("model_metadata"))
    url = f"{api_url}{ENDPOINT_SUBMISSIONS}"
    try:
        response = requests.post(
            url, json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=HTTP_TIMEOUT_POST_S,
        )
    except requests.RequestException as exc:
        raise ConnectionError(f"Could not reach {url}: {exc}") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(
            f"Submission rejected ({response.status_code}): {detail}"
        )
    result = response.json()
    logger.info("Server response: %s", result)
    return result
