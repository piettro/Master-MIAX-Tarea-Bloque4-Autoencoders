"""Central configuration: paths, API settings, hyper-parameters, constants.

Every magic number used by the pipeline lives here so that experiments are
reproducible and easy to audit. Values that are part of the competition
contract (seed, selection parameters, annualisation factor) must NOT be
changed, otherwise the server rejects the submission.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

METADATA_FILENAME = "metadata.json"
PRICES_FILENAME = "prices.parquet"
BENCHMARK_FILENAME = "benchmark.parquet"

# --------------------------------------------------------------------------- #
#  Web-app API (data download + submissions)
# --------------------------------------------------------------------------- #
DEFAULT_API_URL = "https://miaxtallerautoencoders-production.up.railway.app"
API_URL_ENV_VAR = "MIAX_AE_API_URL"
# NOTE: improved over professor's solution — the group token is read from the
# environment (or a local .env file) instead of being pasted into the code,
# so it is never committed to version control.
TOKEN_ENV_VAR = "MIAX_AE_TOKEN"
HTTP_TIMEOUT_GET_S = 60
HTTP_TIMEOUT_POST_S = 120

ENDPOINT_METADATA = "/api/metadata"
ENDPOINT_PRICES = "/api/prices"
ENDPOINT_BENCHMARK = "/api/benchmark"
ENDPOINT_SUBMISSIONS = "/api/submissions"

# --------------------------------------------------------------------------- #
#  Competition contract (identical on server and client)
# --------------------------------------------------------------------------- #
# Official seed: the server rejects any submission trained with another one.
SEED = 42
TRADING_DAYS_PER_YEAR = 252
DEFAULT_N_ASSETS = 25
BENCHMARK_NAME = "RSP"
DEFAULT_BENCHMARK_COLUMN = "rsp_adj"
# Irreducible TE floor of the equal-weight universe vs RSP (measured on the
# server, validation window). Used only for reporting.
TE_FLOOR_REFERENCE = 0.02

# --------------------------------------------------------------------------- #
#  PCA baseline (the "ghost competitor" of the leaderboard)
# --------------------------------------------------------------------------- #
PCA_COMPONENTS = 3
PCA_RANDOM_STATE = 0


# --------------------------------------------------------------------------- #
#  Autoencoder hyper-parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DenoisingAEConfig:
    """Hyper-parameters of the reference denoising autoencoder.

    Defaults reproduce the professor's solution: Gaussian input noise, one
    ``tanh`` hidden layer, a small *linear* bottleneck and a symmetric
    decoder with a linear reconstruction layer.

    Attributes:
        latent_dim: Size of the bottleneck (number of latent factors).
        hidden_units: Width of the hidden layer on each side.
        hidden_activation: Activation of the hidden layers.
        noise_stddev: Std. dev. of the Gaussian noise added to the inputs
            during training (denoising regularisation).
        dropout_rate: Optional dropout after each hidden layer (``0.0``
            disables it, as in the professor's solution).
    """

    latent_dim: int = 8
    hidden_units: int = 64
    hidden_activation: str = "tanh"
    noise_stddev: float = 0.3
    dropout_rate: float = 0.0


@dataclass(frozen=True)
class BaselineAEConfig:
    """Hyper-parameters of the original (uncorrected) notebook skeleton.

    Kept only to reproduce the original submission and show why it fails:
    a single ``relu`` layer straight into a 400-dimensional latent space
    memorises the training data instead of compressing it.

    Attributes:
        latent_dim: Size of the (far too large) latent space.
        latent_activation: Activation of the latent layer.
    """

    latent_dim: int = 400
    latent_activation: str = "relu"


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyper-parameters fixed by the workshop scaffold.

    Attributes:
        epochs: Maximum number of epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        loss: Reconstruction loss.
        validation_split: Fraction of rows (stocks) held out for early
            stopping. Keras takes the *last* rows, without shuffling.
        early_stopping_patience: Epochs without ``val_loss`` improvement
            before stopping (best weights are restored).
    """

    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    loss: str = "mse"
    validation_split: float = 0.1
    early_stopping_patience: int = 15


# --------------------------------------------------------------------------- #
#  Visualisation / explainability
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlotConfig:
    """Plotting constants.

    Attributes:
        dpi: Resolution of the saved figures.
        tsne_perplexity: t-SNE neighbourhood size.
        color_clip_percentiles: Percentiles used to clip colour scales so a
            single outlier does not saturate the colour map.
        heatmap_abs_threshold: Correlation magnitude above which heatmap
            annotations are drawn in white.
    """

    dpi: int = 120
    tsne_perplexity: float = 30.0
    color_clip_percentiles: tuple[float, float] = (2.0, 98.0)
    heatmap_abs_threshold: float = 0.5


# --------------------------------------------------------------------------- #
#  Top-level run configuration
# --------------------------------------------------------------------------- #
MODEL_DENOISING = "denoising"
MODEL_BASELINE = "baseline"
AVAILABLE_MODELS = (MODEL_DENOISING, MODEL_BASELINE)


@dataclass(frozen=True)
class RunConfig:
    """Everything a single pipeline run needs.

    Attributes:
        model: Which autoencoder to train (``"denoising"`` or
            ``"baseline"``).
        denoising: Denoising AE hyper-parameters.
        baseline: Original skeleton hyper-parameters.
        training: Training hyper-parameters.
        plots: Plotting constants.
        seed: Global random seed (must stay at the official value to
            submit).
        n_assets: Number of stocks in the tracker. ``None`` means "use the
            value published in the competition metadata".
        make_plots: Whether to render and save figures.
        results_dir: Where metrics, tracker and figures are written.
    """

    model: str = MODEL_DENOISING
    denoising: DenoisingAEConfig = field(default_factory=DenoisingAEConfig)
    baseline: BaselineAEConfig = field(default_factory=BaselineAEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    seed: int = SEED
    n_assets: int | None = None
    make_plots: bool = True
    results_dir: Path = RESULTS_DIR

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If the model name is unknown or ``n_assets`` is not
                positive.
        """
        if self.model not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model '{self.model}'. "
                f"Choose one of {AVAILABLE_MODELS}."
            )
        if self.n_assets is not None and self.n_assets < 1:
            raise ValueError(
                f"n_assets must be positive, got {self.n_assets}."
            )

    @property
    def latent_dim(self) -> int:
        """Latent dimension of the selected model."""
        if self.model == MODEL_DENOISING:
            return self.denoising.latent_dim
        return self.baseline.latent_dim


def get_api_url() -> str:
    """Return the API base URL (environment override or default).

    Returns:
        The base URL without a trailing slash.
    """
    return os.environ.get(API_URL_ENV_VAR, DEFAULT_API_URL).rstrip("/")


def get_token() -> str | None:
    """Return the group token from the environment, if any.

    A ``.env`` file at the project root is loaded first (if
    ``python-dotenv`` is installed), so the token can be kept out of the
    shell history as well as out of the code.

    Returns:
        The token, or ``None`` when it is not configured.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional convenience
        pass
    else:
        load_dotenv(PROJECT_ROOT / ".env")
    token = os.environ.get(TOKEN_ENV_VAR, "").strip()
    return token or None
