"""Unit tests for src.submission.client (no network access)."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from src.models.autoencoder import build_denoising_autoencoder
from src.submission import client


def _payload(seed=42, is_final=False):
    return client.build_payload(
        ["A", "B"], np.zeros((2, 3)), ["A"], [1.0], is_final=is_final,
        model_metadata={"seed": seed},
    )


def test_payload_structure():
    p = _payload()
    assert p["latent_space"]["tickers"] == ["A", "B"]
    assert np.asarray(p["latent_space"]["coordinates"]).shape == (2, 3)
    assert p["tracker"] == {"tickers": ["A"], "weights": [1.0]}
    assert p["is_final_submission"] is False
    assert p["request_test_evaluation"] is False


def test_payload_shape_mismatch():
    with pytest.raises(ValueError):
        client.build_payload(["A"], np.zeros((2, 3)), ["A"], [1.0])
    with pytest.raises(ValueError):
        client.build_payload(["A"], np.zeros((1, 3)), ["A"], [0.5, 0.5])


def test_final_without_confirmation_is_blocked():
    with mock.patch.object(client.requests, "post") as post:
        with pytest.raises(ValueError, match="irreversible"):
            client.submit("http://x", "tok", _payload(is_final=True))
        post.assert_not_called()


@pytest.mark.parametrize("seed", [None, 7, "abc"])
def test_wrong_seed_is_blocked(seed):
    with mock.patch.object(client.requests, "post") as post:
        with pytest.raises(ValueError, match="SEED"):
            client.submit("http://x", "tok", _payload(seed=seed))
        post.assert_not_called()


def test_rejection_raises_runtime_error():
    resp = mock.Mock(status_code=409, text="closed")
    resp.json.return_value = {"detail": "competition closed"}
    with mock.patch.object(client.requests, "post", return_value=resp):
        with pytest.raises(RuntimeError, match="competition closed"):
            client.submit("http://x", "tok", _payload())


def test_successful_submission():
    resp = mock.Mock(status_code=200)
    resp.json.return_value = {"status": "validated"}
    with mock.patch.object(client.requests, "post",
                           return_value=resp) as post:
        assert client.submit("http://x", "tok", _payload()) == \
            {"status": "validated"}
    assert post.call_args.kwargs["headers"] == {
        "Authorization": "Bearer tok"}


def test_architecture_summary():
    model, _ = build_denoising_autoencoder(20)
    arch = client.architecture_summary(model)
    assert arch["n_params"] == model.count_params()
    noise = next(layer for layer in arch["layers"]
                 if layer["type"] == "GaussianNoise")
    assert noise["stddev"] == pytest.approx(0.3)
