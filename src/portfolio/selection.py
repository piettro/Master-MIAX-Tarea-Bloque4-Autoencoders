"""Tracker selection procedure — CORE OF THE COMPETITION CONTRACT.

WARNING: the *logic* of this module must stay identical to the server's
``webapp/selection/selection.py``. It is an English port of the scaffold
``utils/selection.py`` (kept verbatim in ``professor_solution/utils/``);
only comments, docstrings and error messages were translated. The server
re-runs this exact procedure on the submitted latent space and rejects the
submission if tickers/weights do not match.

Algorithm (version 1.1.x):

1. Take the latent matrix ``X`` of shape ``(N, latent_dim)`` in the same
   order as ``tickers``.
2. **Project onto the unit sphere (cosine distance)**: divide each row by
   its Euclidean norm, keeping only the *direction* of the embedding. Stocks
   are grouped by *how* they co-move (the shape of their exposure to common
   factors), not by how large their moves are.
3. Run **KMeans** with ``n_clusters = n_assets``, fixed seed, ``n_init``
   and ``"lloyd"`` on the normalised vectors (*spherical k-means*: on the
   unit sphere, Euclidean distance is a monotone function of cosine
   similarity). Fully deterministic for a given input.
4. **One representative per cluster — its medoid**: the stock with the
   highest total cosine similarity to the rest of its cluster, i.e. the
   most central *real* asset (not the abstract centroid). Ties are broken
   by the lowest original index.
5. **Weights by cluster size**: ``w_c = |cluster_c| / N`` (sum to 1).
6. The output is sorted **alphabetically by ticker** (canonical order).
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------- #
#  Fixed procedure parameters. Changing them breaks the contract with the
#  server: bump the version and regenerate notebook + server together.
#  (Deliberately not moved to config.py: they are part of the contract.)
# --------------------------------------------------------------------------- #
SELECTION_VERSION = "1.1.1"
SELECTION_SEED = 42
KMEANS_N_INIT = 10
KMEANS_ALGORITHM = "lloyd"
SELECTION_METRIC = "cosine"
DEFAULT_N_ASSETS = 25
SIGNATURE_DIGEST_CHARS = 16


def select_tracker(
    coordinates: Sequence[Sequence[float]] | np.ndarray,
    tickers: Sequence[str],
    n_assets: int = DEFAULT_N_ASSETS,
    seed: int = SELECTION_SEED,
) -> tuple[list[str], list[float]]:
    """Derive the tracker (tickers + weights) from the latent space.

    Args:
        coordinates: Latent matrix ``(N, latent_dim)`` in the same order as
            ``tickers``.
        tickers: The ``N`` tickers of the universe.
        n_assets: Number of stocks in the tracker (default 25).
        seed: KMeans seed (default ``SELECTION_SEED``).

    Returns:
        ``(selected_tickers, weights)`` sorted alphabetically by ticker;
        ``weights`` sum to 1.0.

    Raises:
        ValueError: If shapes do not match, tickers are duplicated, or
            ``n_assets`` distinct representatives cannot be formed
            (degenerate latent space). Zero-norm rows are NOT an error:
            they are treated as the origin and are never selected.
    """
    X = np.asarray(coordinates, dtype=np.float64)
    tickers = list(tickers)

    if X.ndim != 2:
        raise ValueError(
            "`coordinates` must be 2-D (N, latent_dim); "
            f"got ndim={X.ndim}."
        )
    n_samples = X.shape[0]
    if n_samples != len(tickers):
        raise ValueError(
            f"Latent rows ({n_samples}) != number of tickers "
            f"({len(tickers)})."
        )
    if len(set(tickers)) != len(tickers):
        raise ValueError("The universe contains duplicated tickers.")
    if n_assets < 1 or n_assets > n_samples:
        raise ValueError(
            f"n_assets ({n_assets}) must be in [1, N]; N={n_samples}."
        )

    # Cosine distance: project every stock onto the unit sphere. A zero-norm
    # row (e.g. a relu latent collapsing to 0) has no direction: it stays at
    # the origin (norm treated as 1) and scores 0 as a medoid candidate.
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
                f"Cluster {c} is empty: degenerate latent space (collinear "
                "or duplicated points). Cannot form distinct representatives."
            )
        # Representative = MEDOID: the member with the highest TOTAL cosine
        # similarity to the rest of its cluster. Vectors are unit-norm, so
        # cosine similarity is the dot product. Ties -> lowest original
        # index (np.argmax returns the first maximum).
        members = Xn[member_idx]
        total_sim = (members @ members.T).sum(axis=1) - 1.0  # drop self-sim
        rep_idx = int(member_idx[int(np.argmax(total_sim))])
        used_indices.add(rep_idx)
        weight = member_idx.size / n_samples
        selected.append((tickers[rep_idx], weight))

    if len(used_indices) != n_assets:
        raise ValueError(
            "Representatives are not distinct across clusters; "
            "degenerate latent space."
        )

    # Canonical order by ticker.
    selected.sort(key=lambda t: t[0])
    out_tickers = [t for t, _ in selected]
    out_weights = [w for _, w in selected]
    return out_tickers, out_weights


def selection_signature(n_assets: int = DEFAULT_N_ASSETS) -> str:
    """Signature of the procedure (version + parameters).

    Lets the server and the client check that they share exactly the same
    selection logic before trusting the coherence verification.

    Args:
        n_assets: Number of stocks in the tracker.

    Returns:
        ``"<version>:<16-hex-digest>"``.
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
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{SELECTION_VERSION}:{digest[:SIGNATURE_DIGEST_CHARS]}"
