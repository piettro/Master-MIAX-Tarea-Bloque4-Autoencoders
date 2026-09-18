"""Linear PCA baseline — the leaderboard's "ghost competitor".

PCA is, in essence, a linear autoencoder: with linear encoder/decoder and
an MSE loss an autoencoder recovers the same subspace. A non-linear
autoencoder only adds value if its tracker's TE is *below* PCA's.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from src.utils.config import PCA_COMPONENTS, PCA_RANDOM_STATE


def pca_latent(
    x_train: np.ndarray,
    n_components: int = PCA_COMPONENTS,
    random_state: int = PCA_RANDOM_STATE,
) -> np.ndarray:
    """Project each stock onto the first principal components.

    Args:
        x_train: Scaled training matrix ``(n_stocks, n_days)``.
        n_components: Number of components (latent size).
        random_state: Seed of the randomized SVD solver (if used).

    Returns:
        Latent coordinates ``(n_stocks, n_components)``.

    Raises:
        ValueError: If ``n_components`` exceeds the matrix rank bound.
    """
    max_components = min(x_train.shape)
    if not 1 <= n_components <= max_components:
        raise ValueError(
            f"n_components must be in [1, {max_components}], "
            f"got {n_components}."
        )
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(x_train)
