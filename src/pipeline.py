"""End-to-end pipeline: returns -> autoencoder -> latent -> tracker -> TE.

Only step 1 (the autoencoder) is free in the competition; latent
projection, clustering/selection and TE measurement are fixed and identical
for every team.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from src.data.dataset import MarketDataset
from src.data.preprocessing import asset_return_matrix, fit_scaler, to_returns
from src.evaluation.explainability import (
    company_features,
    latent_feature_correlation,
)
from src.evaluation.metrics import PerformanceSummary, summarize
from src.evaluation.tracker import TrackerEvaluation, evaluate_tracker
from src.models.autoencoder import (
    build_baseline_autoencoder,
    build_denoising_autoencoder,
)
from src.models.pca_baseline import pca_latent
from src.models.training import TrainingResult, encode, train_autoencoder
from src.portfolio.backtest import equal_weights
from src.portfolio.selection import select_tracker, selection_signature
from src.utils.config import (
    MODEL_DENOISING,
    PCA_COMPONENTS,
    TE_FLOOR_REFERENCE,
    RunConfig,
)
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)

AE_LABEL = "Autoencoder"
PCA_LABEL = "PCA (baseline)"
EW_LABEL = "EW universe"


@dataclass
class PipelineResult:
    """Artefacts of a pipeline run.

    Attributes:
        config: Configuration used.
        tickers: Universe tickers in latent order.
        latent: Autoencoder latent coordinates ``(N, latent_dim)``.
        autoencoder: Trained autoencoder (needed for the submission).
        training: Training summary.
        autoencoder_eval: Evaluation of the autoencoder tracker.
        pca_eval: Evaluation of the PCA tracker.
        ew_floor: EW-universe vs RSP metrics on validation (TE floor).
        latent_feature_corr: Latent-dimension/feature correlations.
        features: Company characteristics used for explainability.
    """

    config: RunConfig
    tickers: list[str]
    latent: np.ndarray
    autoencoder: keras.Model
    training: TrainingResult
    autoencoder_eval: TrackerEvaluation
    pca_eval: TrackerEvaluation
    ew_floor: PerformanceSummary
    latent_feature_corr: pd.DataFrame
    features: pd.DataFrame

    @property
    def beats_pca(self) -> bool:
        """Whether the AE tracker has a lower validation TE than PCA."""
        return (self.autoencoder_eval.te_validation
                < self.pca_eval.te_validation)

    def comparison_frame(self) -> pd.DataFrame:
        """TE on train and validation for the AE and PCA trackers."""
        rows = {
            AE_LABEL: self.autoencoder_eval,
            PCA_LABEL: self.pca_eval,
        }
        return pd.DataFrame(
            {
                "TE train": {k: f"{v.te_train:.2%}" for k, v in rows.items()},
                "TE validation": {k: f"{v.te_validation:.2%}"
                                  for k, v in rows.items()},
            }
        )


def build_model(
    config: RunConfig, input_dim: int
) -> tuple[keras.Model, keras.Model]:
    """Instantiate the autoencoder chosen in the configuration.

    Args:
        config: Run configuration.
        input_dim: Number of input features.

    Returns:
        ``(autoencoder, encoder)``.
    """
    if config.model == MODEL_DENOISING:
        return build_denoising_autoencoder(input_dim, config.denoising)
    return build_baseline_autoencoder(input_dim, config.baseline)


def run_pipeline(dataset: MarketDataset, config: RunConfig) -> PipelineResult:
    """Run the full workshop pipeline on ``dataset``.

    Args:
        dataset: Downloaded market data.
        config: Run configuration.

    Returns:
        All artefacts of the run.
    """
    set_global_seed(config.seed)
    train_w, val_w = dataset.train_window, dataset.validation_window
    n_assets = config.n_assets or dataset.n_assets
    prices = dataset.prices_wide
    bench_returns = to_returns(dataset.benchmark)
    logger.info(
        "Universe: %d stocks | train %s..%s | validation %s..%s | "
        "tracker size %d | selection %s",
        prices.shape[1], train_w.start.date(), train_w.end.date(),
        val_w.start.date(), val_w.end.date(), n_assets,
        selection_signature(n_assets),
    )

    # 1) Irreducible floor: equal-weight universe vs RSP.
    ew_returns = evaluate_tracker(
        EW_LABEL, equal_weights(list(prices.columns)), prices,
        bench_returns, train_w, val_w,
    ).returns
    ew_floor = summarize(val_w.slice(ew_returns), val_w.slice(bench_returns))
    logger.info("EW universe vs RSP, validation TE: %.2f%%",
                100 * ew_floor.tracking_error)

    # 2) Input matrix: one row per stock, train returns, per-day z-score
    #    fitted on train only (no look-ahead).
    x_raw, tickers = asset_return_matrix(prices, train_w)
    _, x_train = fit_scaler(x_raw)
    logger.info("X_train shape %s (stocks x train days)", x_train.shape)

    # 3) PCA baseline ("ghost competitor").
    pca_tickers, pca_weights = select_tracker(
        pca_latent(x_train, PCA_COMPONENTS), tickers, n_assets=n_assets
    )
    pca_eval = evaluate_tracker(
        PCA_LABEL, dict(zip(pca_tickers, pca_weights)), prices,
        bench_returns, train_w, val_w,
    )
    logger.info("PCA TE train %.2f%% | validation %.2f%%",
                100 * pca_eval.te_train, 100 * pca_eval.te_validation)

    # 4) Autoencoder: train, encode, derive tracker with the fixed rule.
    autoencoder, encoder = build_model(config, x_train.shape[1])
    training = train_autoencoder(autoencoder, x_train, config.training)
    latent = encode(encoder, x_train)
    ae_tickers, ae_weights = select_tracker(
        latent, tickers, n_assets=n_assets
    )
    ae_eval = evaluate_tracker(
        AE_LABEL, dict(zip(ae_tickers, ae_weights)), prices,
        bench_returns, train_w, val_w,
    )
    logger.info("AE  TE train %.2f%% | validation %.2f%%",
                100 * ae_eval.te_train, 100 * ae_eval.te_validation)

    # 5) Explainability (train-window characteristics only).
    features = company_features(prices, tickers, bench_returns, train_w)
    corr = latent_feature_correlation(latent, features)

    return PipelineResult(
        config=config,
        tickers=tickers,
        latent=latent,
        autoencoder=autoencoder,
        training=training,
        autoencoder_eval=ae_eval,
        pca_eval=pca_eval,
        ew_floor=ew_floor,
        latent_feature_corr=corr,
        features=features,
    )


def report(result: PipelineResult) -> None:
    """Log a human-readable summary of the run.

    Args:
        result: Pipeline artefacts.
    """
    ae, pca = result.autoencoder_eval, result.pca_eval
    logger.info("Tracker (%d stocks):\n%s", len(ae.weights),
                pd.Series(ae.weights).map("{:.2%}".format).to_string())
    logger.info("Tracker vs PCA - TE against RSP:\n%s",
                result.comparison_frame().to_string())
    if result.beats_pca:
        logger.info("Beats PCA on validation by %.2f pp.",
                    100 * (pca.te_validation - ae.te_validation))
    else:
        logger.warning("Does NOT beat PCA on validation (%.2f%% vs %.2f%%).",
                       100 * ae.te_validation, 100 * pca.te_validation)
    logger.info("Tracker vs RSP - validation:\n%s",
                ae.validation_summary.to_frame(AE_LABEL).to_string())
    logger.info("Irreducible floor (EW universe) ~%.2f%% (server reference "
                "~%.0f%%).", 100 * result.ew_floor.tracking_error,
                100 * TE_FLOOR_REFERENCE)


def save_results(result: PipelineResult, out_dir: Path) -> Path:
    """Persist metrics, tracker and latent space to ``out_dir``.

    Args:
        result: Pipeline artefacts.
        out_dir: Destination directory.

    Returns:
        Path of the metrics JSON file.

    Raises:
        OSError: If the files cannot be written.
    """
    ae = result.autoencoder_eval
    cfg = result.config
    metrics = {
        "model": cfg.model,
        "latent_dim": int(result.latent.shape[1]),
        "seed": cfg.seed,
        "n_params": int(result.autoencoder.count_params()),
        "training": {
            "epochs_run": result.training.epochs_run,
            "best_epoch": result.training.best_epoch,
            "best_val_loss": result.training.best_val_loss,
            "stopped_early": result.training.stopped_early,
        },
        "autoencoder": {
            "te_train": ae.te_train,
            "te_validation": ae.te_validation,
            "validation": ae.validation_summary.to_dict(),
        },
        "pca": {
            "te_train": result.pca_eval.te_train,
            "te_validation": result.pca_eval.te_validation,
            "validation": result.pca_eval.validation_summary.to_dict(),
        },
        "ew_universe_validation": result.ew_floor.to_dict(),
        "beats_pca_validation": result.beats_pca,
    }
    metrics_path = out_dir / "metrics.json"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2),
                                encoding="utf-8")
        pd.Series(ae.weights, name="weight").rename_axis("ticker").to_csv(
            out_dir / "tracker.csv"
        )
        pd.DataFrame(
            result.latent, index=pd.Index(result.tickers, name="ticker"),
            columns=[f"z{d}" for d in range(result.latent.shape[1])],
        ).to_csv(out_dir / "latent.csv")
        result.latent_feature_corr.to_csv(
            out_dir / "latent_feature_correlation.csv"
        )
    except OSError as exc:
        raise OSError(f"Could not write results to {out_dir}: {exc}") from exc
    logger.info("Results written to %s", out_dir)
    return metrics_path


def save_figures(
    result: PipelineResult, dataset: MarketDataset, figures_dir: Path
) -> list[Path]:
    """Render and save every figure of the run.

    Args:
        result: Pipeline artefacts.
        dataset: Market data (for sectors and the benchmark).
        figures_dir: Destination directory.

    Returns:
        The written file paths.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless rendering
    from src.visualization import plots

    cfg = result.config
    dpi = cfg.plots.dpi
    selected = list(result.autoencoder_eval.weights)
    val_w = dataset.validation_window
    figures = {
        "training_loss.png": plots.plot_training_history(
            result.training.history, result.training.best_epoch
        ),
        "latent_tsne_by_sector.png": plots.plot_latent_tsne_by_sector(
            result.latent, result.tickers, selected, dataset.sectors,
            cfg.seed, cfg.plots,
        ),
        "latent_feature_gradients.png": plots.plot_latent_feature_gradients(
            result.latent, result.tickers, selected, result.features,
            cfg.plots,
        ),
        "latent_feature_heatmap.png": plots.plot_latent_feature_heatmap(
            result.latent_feature_corr, cfg.plots
        ),
        "validation_cumulative_returns.png": plots.plot_cumulative_returns(
            {
                AE_LABEL: val_w.slice(result.autoencoder_eval.returns),
                PCA_LABEL: val_w.slice(result.pca_eval.returns),
                dataset.benchmark.name: val_w.slice(
                    to_returns(dataset.benchmark)
                ),
            },
            "Validation window - growth of 1",
        ),
    }
    if result.latent.shape[1] >= 2:
        figures["latent_scatter.png"] = plots.plot_latent_scatter(
            result.latent, result.tickers, selected
        )
    return [plots.save_figure(fig, figures_dir / name, dpi)
            for name, fig in figures.items()]
