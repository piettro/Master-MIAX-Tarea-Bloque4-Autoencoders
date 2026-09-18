"""Entry point: Deep Portfolios with Autoencoders (MIAX workshop).

Examples:
    Run the corrected (denoising) model::

        python main.py

    Reproduce the original, uncorrected submission::

        python main.py --model baseline

    Submit the result to the validation leaderboard::

        python main.py --submit
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

SUPPORTED_PYTHON = ((3, 11), (3, 13))
if not (SUPPORTED_PYTHON[0] <= sys.version_info[:2] <= SUPPORTED_PYTHON[1]):
    sys.exit(
        f"Python {sys.version_info.major}.{sys.version_info.minor} is not "
        "supported: use Python 3.11-3.13 (TensorFlow 2.21 / pandas 3). "
        "On Windows: py -3.13 -m venv .venv"
    )

# Silence TensorFlow's C++ start-up banner before it is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.data.api_client import load_dataset  # noqa: E402
from src.pipeline import (  # noqa: E402
    PipelineResult,
    report,
    run_pipeline,
    save_figures,
    save_results,
)
from src.utils.config import (  # noqa: E402
    AVAILABLE_MODELS,
    DATA_DIR,
    MODEL_DENOISING,
    RESULTS_DIR,
    TOKEN_ENV_VAR,
    RunConfig,
    get_api_url,
    get_token,
)
from src.utils.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger("main")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train an autoencoder, derive a 25-stock tracker of the "
                    "S&P 500 Equal Weight (RSP) and measure its tracking "
                    "error against PCA."
    )
    parser.add_argument("--model", choices=AVAILABLE_MODELS,
                        default=MODEL_DENOISING,
                        help="Autoencoder to train (default: %(default)s).")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="Override the latent dimension of the model.")
    parser.add_argument("--refresh-data", action="store_true",
                        help="Re-download data even if it is cached.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Data cache directory (default: %(default)s).")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Output directory (default: %(default)s).")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip figure rendering.")
    parser.add_argument("--submit", action="store_true",
                        help="Submit to the VALIDATION leaderboard.")
    parser.add_argument("--final", action="store_true",
                        help="Make the submission the IRREVERSIBLE final "
                             "test evaluation (needs --confirm-final).")
    parser.add_argument("--confirm-final", action="store_true",
                        help="Confirm the irreversible final submission.")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    if args.final and not args.submit:
        parser.error("--final requires --submit.")
    if args.latent_dim is not None and args.latent_dim < 1:
        parser.error("--latent-dim must be positive.")
    return args


def build_config(args: argparse.Namespace) -> RunConfig:
    """Translate CLI arguments into a :class:`RunConfig`.

    Args:
        args: Parsed arguments.

    Returns:
        The run configuration.
    """
    config = RunConfig(model=args.model, make_plots=not args.no_plots,
                       results_dir=args.results_dir)
    if args.latent_dim is not None:
        if config.model == MODEL_DENOISING:
            config = replace(config, denoising=replace(
                config.denoising, latent_dim=args.latent_dim))
        else:
            config = replace(config, baseline=replace(
                config.baseline, latent_dim=args.latent_dim))
    return config


def submit_result(result: PipelineResult, api_url: str, token: str | None,
                  final: bool, confirm_final: bool) -> None:
    """Send the run to the web-app.

    Args:
        result: Pipeline artefacts.
        api_url: API base URL.
        token: Group token.
        final: Whether this is the irreversible test submission.
        confirm_final: Explicit confirmation for the final submission.

    Raises:
        ValueError: If no token is configured.
    """
    from src.submission.client import (
        architecture_summary,
        build_payload,
        submit,
    )

    if not token:
        raise ValueError(f"Set {TOKEN_ENV_VAR} to submit.")
    arch = architecture_summary(result.autoencoder)
    ae = result.autoencoder_eval
    payload = build_payload(
        result.tickers, result.latent, list(ae.weights),
        list(ae.weights.values()), is_final=final,
        model_metadata={
            "latent_dim": int(result.latent.shape[1]),
            "seed": result.config.seed,
            "n_params": arch["n_params"],
            "architecture": arch["layers"],
        },
    )
    logger.info("Submitting (%s)...", "FINAL/test" if final else "validation")
    submit(api_url, token, payload, confirm_final=confirm_final)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline from the command line.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    configure_logging(args.log_level)
    config = build_config(args)
    api_url, token = get_api_url(), get_token()
    try:
        dataset = load_dataset(api_url, token, args.data_dir,
                               refresh=args.refresh_data)
        result = run_pipeline(dataset, config)
        report(result)
        save_results(result, config.results_dir)
        if config.make_plots:
            save_figures(result, dataset, config.results_dir / "figures")
        if args.submit:
            submit_result(result, api_url, token, args.final,
                          args.confirm_final)
    except (FileNotFoundError, PermissionError, ConnectionError,
            ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
