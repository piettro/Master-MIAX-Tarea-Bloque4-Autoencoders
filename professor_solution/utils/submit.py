"""Cliente de envío de submissions a la webapp (andamiaje del notebook).

Construye el JSON del contrato (espec. 4.3) y lo manda al endpoint de ingestión,
autenticando con el token de grupo. Gestiona la respuesta (TE, presupuesto de
test restante) y los errores (coherencia, formato).

ATENCIÓN: ``is_final=True`` dispara la evaluación **final irreversible** que
congela el modelo. La función pide confirmación explícita por seguridad.
"""

from __future__ import annotations

import numpy as np
import requests

TIMEOUT = 120

# Semilla oficial de la práctica: todos entrenáis con la misma para que la competición
# sea reproducible y justa. Si se cambia, la entrega se rechaza (aquí y en el servidor).
REQUIRED_SEED = 42


def architecture_summary(model) -> dict:
    """Resumen de la arquitectura de un modelo Keras para enviarlo con la submission.

    Devuelve la lista de capas (tipo, unidades, activación, etc.) y el nº de
    parámetros, para que en el leaderboard se vea qué red entrenó cada equipo.
    """
    layers_info = []
    for lyr in model.layers:
        cfg = lyr.get_config()
        info = {
            "name": lyr.name,
            "type": type(lyr).__name__,
            "units": cfg.get("units"),
            "activation": cfg.get("activation"),
            "rate": cfg.get("rate"),          # Dropout
            "stddev": cfg.get("stddev"),      # GaussianNoise
        }
        layers_info.append({k: v for k, v in info.items() if v is not None})
    try:
        n_params = int(model.count_params())
    except Exception:
        n_params = None
    return {"layers": layers_info, "n_params": n_params}


def build_payload(
    latent_tickers,
    latent_coordinates,
    tracker_tickers,
    tracker_weights,
    competition_id=None,
    model_metadata=None,
    is_final=False,
) -> dict:
    return {
        "competition_id": competition_id,
        "model_metadata": model_metadata or {},
        "latent_space": {
            "tickers": list(latent_tickers),
            "coordinates": np.asarray(latent_coordinates, dtype=float).tolist(),
        },
        "tracker": {
            "tickers": list(tracker_tickers),
            "weights": np.asarray(tracker_weights, dtype=float).tolist(),
        },
        "request_test_evaluation": False,
        "is_final_submission": bool(is_final),
    }


def submit(
    api_url: str,
    token: str,
    latent_tickers,
    latent_coordinates,
    tracker_tickers,
    tracker_weights,
    is_final: bool = False,
    model_metadata=None,
    competition_id=None,
    confirm_final: bool = False,
) -> dict:
    """Envía la submission. Devuelve la respuesta del servidor (incluye el TE).

    - Envío normal → evalúa en **validación** (leaderboard en vivo). Ilimitado.
    - ``is_final=True`` evalúa en **test** de forma **irreversible** y congela el
      modelo (única evaluación de test); requiere ``confirm_final=True``.
    """
    if is_final and not confirm_final:
        raise ValueError(
            "Evaluación FINAL irreversible: congela tu modelo y no admite cambios. "
            "Si estás seguro, vuelve a llamar con confirm_final=True."
        )

    # La semilla debe ser la oficial (42): reproducibilidad y juego limpio.
    seed = (model_metadata or {}).get("seed")
    try:
        seed_ok = seed is not None and int(seed) == REQUIRED_SEED
    except (TypeError, ValueError):
        seed_ok = False
    if not seed_ok:
        raise ValueError(
            f"La semilla SEED debe ser {REQUIRED_SEED}: la competición es reproducible y todos "
            f"entrenáis con la misma semilla. Tienes seed={seed!r}. Restaura SEED = {REQUIRED_SEED}, "
            f"vuelve a entrenar y reenvía. (El servidor también lo rechaza.)"
        )

    payload = build_payload(
        latent_tickers, latent_coordinates, tracker_tickers, tracker_weights,
        competition_id=competition_id, model_metadata=model_metadata, is_final=is_final,
    )
    r = requests.post(
        f"{api_url}/api/submissions",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=TIMEOUT,
    )
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"Envío rechazado ({r.status_code}): {detail}")
    return r.json()
