"""Procedimiento de selección del tracker — NÚCLEO DEL CONTRATO.

⚠️  ESTE FICHERO DEBE SER IDÉNTICO EN SERVIDOR Y NOTEBOOK.
    (``webapp/selection/selection.py`` == ``notebook/utils/selection.py``)

A partir del **espacio latente** que produce el autoencoder del alumno (una
coordenada por acción del universo), este módulo deriva de forma **determinista**
las ``n`` acciones del tracker y sus pesos. El servidor re-ejecuta exactamente
este mismo procedimiento para *verificar la coherencia* de la submission
(spec 3.8): si las acciones/pesos enviados no coinciden con los que produce este
código sobre el latente enviado, la entrega se rechaza.

Algoritmo (versión 1.1.0), documentado sin ambigüedad para reproducibilidad:

1. Se toma la matriz de coordenadas latentes ``X`` de forma ``(N, latent_dim)``,
   en el **mismo orden** que la lista ``tickers`` (longitud ``N``).
2. **Normalización a la esfera unidad (distancia coseno)**: cada fila de ``X`` se
   divide por su norma euclídea → vectores unitarios que conservan solo la
   **dirección** del embedding, no su magnitud. Agrupamos las acciones por *cómo*
   co-mueven (la forma de su exposición a los factores comunes), no por cuánto
   oscilan en tamaño; el coseno mide exactamente eso y es la distancia natural
   entre embeddings.
3. Se ejecuta **KMeans** con ``n_clusters = n_assets`` (por defecto 25), semilla
   fija ``SELECTION_SEED``, ``n_init = KMEANS_N_INIT`` y algoritmo ``"lloyd"``
   **sobre los vectores normalizados** (*spherical k-means*: en la esfera unidad,
   la distancia euclídea que minimiza KMeans equivale a ordenar por similitud
   coseno). Todo es determinista dada la misma entrada y los mismos parámetros.
4. **Un representante por cluster (su *medoide*)**: la acción con **mayor similitud
   coseno total al resto de miembros** de su cluster, es decir, el **activo real más
   central** del grupo (un caso concreto que lo resume, no el centroide abstracto).
   Empates se rompen por el **índice original más bajo** (orden de ``tickers``).
5. **Pesos por tamaño de cluster**: ``peso_c = |cluster_c| / N``. Como la suma de
   tamaños de cluster es ``N``, los pesos suman 1 exactamente.
6. La salida se devuelve **ordenada alfabéticamente por ticker** (orden canónico),
   de modo que la comparación servidor↔notebook sea estable.

La tolerancia de coherencia (configurable por competición) debe ser **laxa en los
pesos** (absorbe diferencias de coma flotante) pero **estricta en el conjunto de
tickers** (el conjunto debe coincidir exactamente).
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------- #
#  Parámetros fijos del procedimiento. Cambiarlos altera el contrato:
#  súbele la versión y regenera notebook + servidor a la vez.
# --------------------------------------------------------------------------- #
SELECTION_VERSION = "1.1.1"
SELECTION_SEED = 42
KMEANS_N_INIT = 10
KMEANS_ALGORITHM = "lloyd"
SELECTION_METRIC = "cosine"
DEFAULT_N_ASSETS = 25


def select_tracker(
    coordinates: Sequence[Sequence[float]] | np.ndarray,
    tickers: Sequence[str],
    n_assets: int = DEFAULT_N_ASSETS,
    seed: int = SELECTION_SEED,
) -> tuple[list[str], list[float]]:
    """Deriva el tracker (tickers + pesos) a partir del espacio latente.

    Args:
        coordinates: matriz ``(N, latent_dim)`` de coordenadas latentes, en el
            mismo orden que ``tickers``.
        tickers: lista de los ``N`` tickers del universo.
        n_assets: número de acciones del tracker (por defecto 25).
        seed: semilla de KMeans (por defecto ``SELECTION_SEED``).

    Returns:
        ``(tickers_sel, weights)`` ordenados alfabéticamente por ticker;
        ``weights`` suma 1.0.

    Raises:
        ValueError: si las dimensiones no encajan, hay tickers duplicados, o no
            se pueden formar ``n_assets`` representantes distintos (latente
            degenerado). Las filas de norma cero (latente sin dirección para alguna
            acción) NO son error: se tratan como el origen y no salen elegidas.
    """
    X = np.asarray(coordinates, dtype=np.float64)
    tickers = list(tickers)

    if X.ndim != 2:
        raise ValueError(f"`coordinates` debe ser 2D (N, latent_dim); recibido ndim={X.ndim}.")
    n_samples = X.shape[0]
    if n_samples != len(tickers):
        raise ValueError(
            f"Nº de filas del latente ({n_samples}) != nº de tickers ({len(tickers)})."
        )
    if len(set(tickers)) != len(tickers):
        raise ValueError("Hay tickers duplicados en el universo.")
    if n_assets < 1 or n_assets > n_samples:
        raise ValueError(
            f"n_assets ({n_assets}) debe estar en [1, N]; N={n_samples}."
        )

    # Distancia coseno: normalizamos cada acción a la esfera unidad (solo dirección).
    # Una fila de norma cero (p.ej. un latente relu que colapsa a 0 para alguna acción)
    # no tiene dirección: la dejamos en el origen (norma tratada como 1). No rompe la
    # selección; esa acción no puntúa como medoide (similitud coseno 0) y la distinción
    # de representantes de más abajo detecta la degeneración real si la hubiera.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.where(norms == 0.0, 1.0, norms)

    kmeans = KMeans(
        n_clusters=n_assets,
        random_state=seed,
        n_init=KMEANS_N_INIT,
        algorithm=KMEANS_ALGORITHM,
    )
    labels = kmeans.fit_predict(Xn)

    selected: list[tuple[str, float]] = []
    used_indices: set[int] = set()
    for c in range(n_assets):
        member_idx = np.flatnonzero(labels == c)
        if member_idx.size == 0:
            raise ValueError(
                f"Cluster {c} vacío: el latente es degenerado (puntos colineales o "
                "duplicados). No se pueden formar 25 representantes distintos."
            )
        # Representante = MEDOIDE del grupo: el activo real con mayor similitud coseno
        # TOTAL al resto de sus miembros (el más central). Como los vectores están
        # normalizados, la similitud coseno es el producto escalar. Empate -> índice
        # original más bajo (np.argmax devuelve el primer máximo).
        members = Xn[member_idx]
        total_sim = (members @ members.T).sum(axis=1) - 1.0   # resta la auto-similitud (=1)
        rep_idx = int(member_idx[int(np.argmax(total_sim))])
        used_indices.add(rep_idx)
        weight = member_idx.size / n_samples
        selected.append((tickers[rep_idx], weight))

    if len(used_indices) != n_assets:
        raise ValueError(
            "Representantes no distintos entre clusters; latente degenerado."
        )

    # Orden canónico por ticker.
    selected.sort(key=lambda t: t[0])
    out_tickers = [t for t, _ in selected]
    out_weights = [w for _, w in selected]
    return out_tickers, out_weights


def selection_signature(n_assets: int = DEFAULT_N_ASSETS) -> str:
    """Firma del procedimiento (versión + parámetros).

    Sirve para que servidor y notebook comprueben que comparten exactamente la
    misma lógica de selección antes de confiar en la verificación de coherencia.
    """
    payload = "|".join(
        [
            f"v={SELECTION_VERSION}",
            f"seed={SELECTION_SEED}",
            f"n_init={KMEANS_N_INIT}",
            f"algo={KMEANS_ALGORITHM}",
            f"metric={SELECTION_METRIC}",
            f"n_assets={n_assets}",
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{SELECTION_VERSION}:{digest}"
