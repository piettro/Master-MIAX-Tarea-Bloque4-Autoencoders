"""Plotting helpers. Every function returns the Matplotlib figure.

Figures are rendered with the non-interactive ``Agg`` backend by
:func:`save_figure` callers, so the pipeline runs headless (CI, servers).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.utils.config import PCA_RANDOM_STATE, PlotConfig

logger = logging.getLogger(__name__)

UNKNOWN_SECTOR = "?"
SELECTED_LABEL = "tracker"


def save_figure(fig: Figure, path: Path, dpi: int) -> Path:
    """Save and close a figure.

    Args:
        fig: The figure.
        path: Destination file (parent folders are created).
        dpi: Resolution.

    Returns:
        The written path.

    Raises:
        OSError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    except OSError as exc:
        raise OSError(f"Could not save figure {path}: {exc}") from exc
    finally:
        plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_training_history(
    history: Mapping[str, Sequence[float]], best_epoch: int
) -> Figure:
    """Train / validation reconstruction loss per epoch.

    Args:
        history: ``loss`` and ``val_loss`` lists.
        best_epoch: 1-based epoch whose weights were restored.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["loss"], label="train")
    ax.plot(history["val_loss"], label="validation (hold-out stocks)")
    ax.axvline(best_epoch - 1, color="green", ls="--", lw=1,
               label=f"best (epoch {best_epoch})")
    ax.set_title("Reconstruction loss (MSE)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    return fig


def _selected_indices(
    tickers: Sequence[str], selected: Sequence[str]
) -> list[int]:
    """Row indices of the selected tickers."""
    position = {t: i for i, t in enumerate(tickers)}
    return [position[t] for t in selected]


def plot_latent_scatter(
    latent: np.ndarray, tickers: Sequence[str], selected: Sequence[str]
) -> Figure:
    """First two latent dimensions with the tracker stocks highlighted.

    Args:
        latent: Latent matrix ``(n_stocks, latent_dim)``.
        tickers: Tickers in latent order.
        selected: Tracker tickers.

    Returns:
        The figure.

    Raises:
        ValueError: If the latent space has fewer than 2 dimensions.
    """
    if latent.shape[1] < 2:
        raise ValueError("Need at least 2 latent dimensions to scatter.")
    idx = _selected_indices(tickers, selected)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(latent[:, 0], latent[:, 1], s=10, alpha=0.4,
               label="universe")
    ax.scatter(latent[idx, 0], latent[idx, 1], c="red", s=40,
               label=f"{len(idx)} representatives")
    ax.set_xlabel("latent 0")
    ax.set_ylabel("latent 1")
    ax.set_title("Latent space (first two dimensions)")
    ax.legend()
    return fig


def plot_latent_tsne_by_sector(
    latent: np.ndarray,
    tickers: Sequence[str],
    selected: Sequence[str],
    sectors: Mapping[str, str],
    seed: int,
    config: PlotConfig,
) -> Figure:
    """t-SNE view of the latent space coloured by sector.

    t-SNE is only a lens to *look* at the space: it does not project new
    data and its axes carry no meaning. If the autoencoder captured real
    structure, stocks of the same sector cluster together even though the
    sector was never an input.

    Args:
        latent: Latent matrix.
        tickers: Tickers in latent order.
        selected: Tracker tickers (circled).
        sectors: Mapping ticker -> sector.
        seed: t-SNE random state.
        config: Plot constants.

    Returns:
        The figure.
    """
    # NOTE: added — required by assignment (section 9, explainability) /
    # missing in original submission.
    labels = [sectors.get(t, UNKNOWN_SECTOR) for t in tickers]
    unique = sorted(set(labels))
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(unique), 1)))
    perplexity = min(config.tsne_perplexity, (len(tickers) - 1) / 3)
    points = TSNE(
        n_components=2, init="pca", perplexity=perplexity,
        learning_rate="auto", random_state=seed,
    ).fit_transform(latent)

    fig, ax = plt.subplots(figsize=(9, 6))
    labels_arr = np.asarray(labels)
    for color, sector in zip(palette, unique):
        mask = labels_arr == sector
        ax.scatter(points[mask, 0], points[mask, 1], s=20, alpha=0.75,
                   color=color, label=sector)
    idx = _selected_indices(tickers, selected)
    ax.scatter(points[idx, 0], points[idx, 1], s=140, facecolors="none",
               edgecolors="black", linewidths=1.6,
               label=f"{SELECTED_LABEL} ({len(idx)})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
              frameon=False)
    ax.set_title("Latent space by sector - t-SNE view (o = tracker)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    return fig


def plot_latent_feature_gradients(
    latent: np.ndarray,
    tickers: Sequence[str],
    selected: Sequence[str],
    features: pd.DataFrame,
    config: PlotConfig,
) -> Figure:
    """Linear 2-D projection coloured by company characteristics.

    Unlike t-SNE, PCA-2D preserves the latent axes, so a colour *gradient*
    along an axis means that axis encodes the property.

    Args:
        latent: Latent matrix.
        tickers: Tickers in latent order.
        selected: Tracker tickers (circled).
        features: One column per characteristic, rows in latent order.
        config: Plot constants.

    Returns:
        The figure.
    """
    # NOTE: added — required by assignment (section 9, explainability) /
    # missing in original submission.
    if latent.shape[1] == 2:
        points = latent
    else:
        points = PCA(n_components=2,
                     random_state=PCA_RANDOM_STATE).fit_transform(latent)
    idx = _selected_indices(tickers, selected)
    n_feat = features.shape[1]
    fig, axes = plt.subplots(1, n_feat, figsize=(5.3 * n_feat, 4.6),
                             squeeze=False)
    for ax, name in zip(axes[0], features.columns):
        values = features[name].to_numpy()
        lo, hi = np.nanpercentile(values, config.color_clip_percentiles)
        sc = ax.scatter(points[:, 0], points[:, 1], c=values,
                        cmap="viridis", s=22, alpha=0.85, vmin=lo, vmax=hi)
        ax.scatter(points[idx, 0], points[idx, 1], s=130,
                   facecolors="none", edgecolors="red", linewidths=1.4)
        ax.set_title(name)
        ax.set_xlabel("PC 0 of latent")
        ax.set_ylabel("PC 1 of latent")
        fig.colorbar(sc, ax=ax, shrink=0.85, extend="both")
    fig.suptitle("Latent space by company characteristic (o = tracker)")
    fig.tight_layout()
    return fig


def plot_latent_feature_heatmap(
    correlations: pd.DataFrame, config: PlotConfig
) -> Figure:
    """Heatmap of latent-dimension vs characteristic correlations.

    Args:
        correlations: ``(latent_dim, n_features)`` correlation table.
        config: Plot constants.

    Returns:
        The figure.
    """
    # NOTE: added — required by assignment (section 9, explainability) /
    # missing in original submission.
    values = correlations.to_numpy()
    n_dims, n_feat = values.shape
    fig, ax = plt.subplots(figsize=(6, 0.5 * n_dims + 1.5))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(correlations.columns, rotation=15)
    ax.set_yticks(range(n_dims))
    ax.set_yticklabels(correlations.index)
    for d in range(n_dims):
        for j in range(n_feat):
            strong = abs(values[d, j]) > config.heatmap_abs_threshold
            ax.text(j, d, f"{values[d, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if strong else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    ax.set_title("What does each latent dimension capture?")
    fig.tight_layout()
    return fig


def plot_cumulative_returns(series: Mapping[str, pd.Series],
                            title: str) -> Figure:
    """Growth of 1 unit invested for several return series.

    Args:
        series: Mapping label -> daily returns.
        title: Figure title.

    Returns:
        The figure.
    """
    # NOTE: improved over professor's solution — visual comparison of the
    # trackers against RSP, complementing the TE table.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for label, returns in series.items():
        wealth = (1.0 + returns.fillna(0.0)).cumprod()
        ax.plot(wealth.index, wealth.to_numpy(), label=label)
    ax.set_title(title)
    ax.set_ylabel("growth of 1")
    ax.legend()
    fig.autofmt_xdate()
    return fig
