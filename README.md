# Deep Portfolios with Autoencoders

> Master's in AI & Quantum Computing Applied to Financial Markets (MIAX 14)
> Course: Advanced Artificial Intelligence | Assignment: Autoencoders workshop — sparse S&P 500 Equal Weight tracker

## Overview

The goal is to replicate the **S&P 500 Equal Weight index (ETF RSP)**, about
500 stocks, with a portfolio of **only 25 stocks**, while minimising the
out-of-sample **tracking error**

$$\mathrm{TE} = \sqrt{252}\,\sigma\left(r^{\text{track}}_t - r^{\text{RSP}}_t\right).$$

The stocks are chosen by a neural network, not by hand:

1. An **autoencoder** is trained on the daily returns of the universe
   (342 stocks, train 2015–2021). Each stock is one row (one sample); its
   ~1762 daily returns are the features.
2. The **encoder** maps each stock to a low-dimensional **embedding**
   (latent space).
3. A **fixed** procedure (the same for every team, re-run by the server)
   projects embeddings onto the unit sphere (cosine similarity), runs
   k-means with 25 clusters, picks each cluster's **medoid** (most central
   real stock) and weights it by cluster size.
4. The tracker is backtested with **quarterly rebalancing** (like RSP) and
   its TE is compared with a linear **PCA** baseline (the "ghost
   competitor") and with sector-stratified random "monkey" portfolios.

Concepts covered: undercomplete and denoising autoencoders, the PCA ↔ linear
autoencoder equivalence, embeddings and cosine similarity, spherical k-means
and medoids, train/validation/test discipline (overfitting to validation),
survivorship bias, and latent-space explainability (t-SNE by sector, and
latent axes against volatility/growth/beta).

## Project Structure

```
.
├── main.py                        # single CLI entry point
├── src/
│   ├── pipeline.py                # orchestration, reporting, result/figure export
│   ├── data/
│   │   ├── api_client.py          # authenticated download + parquet cache
│   │   ├── dataset.py             # DateWindow, MarketDataset
│   │   └── preprocessing.py       # wide panel, returns, AE matrix, scaler
│   ├── models/
│   │   ├── autoencoder.py         # denoising AE (corrected) + original baseline
│   │   ├── training.py            # fit with early stopping, encode
│   │   └── pca_baseline.py        # PCA "ghost competitor"
│   ├── portfolio/
│   │   ├── selection.py           # FIXED selection contract (server-identical)
│   │   └── backtest.py            # quarterly-rebalanced portfolio returns
│   ├── evaluation/
│   │   ├── metrics.py             # TE, correlation, CAGR, vol, max drawdown
│   │   ├── tracker.py             # per-window tracker evaluation
│   │   └── explainability.py      # vol/growth/beta, latent correlations
│   ├── visualization/
│   │   └── plots.py               # loss, latent, t-SNE, gradients, heatmap
│   ├── submission/
│   │   └── client.py              # leaderboard submission (opt-in)
│   └── utils/
│       ├── config.py              # all constants and hyper-parameters
│       ├── logging_config.py
│       └── seed.py
├── tests/                         # pytest suite (offline, synthetic market)
├── docs/architecture.md           # Mermaid diagrams
├── notebooks/
│   ├── exploration.ipynb          # interactive companion (uses src/)
│   └── archive/original_submission.ipynb   # original notebook, token redacted
├── professor_solution/            # reference notebook + original scaffold utils/
├── materials/                     # brief, lecture transcript, theory lecture
├── requirements.txt
├── setup.cfg                      # flake8 + pytest configuration
├── .env.example
└── .gitignore
```

## Setup & Installation

Requires **Python 3.11–3.13**. TensorFlow 2.21 has no wheels for Python
3.14, and pandas 3 needs Python ≥ 3.11.

```bash
# 1. Clone the repo
git clone <repo-url>
cd autoencoder

# 2. Create and activate a virtual environment
python -m venv .venv            # Windows with several Pythons: py -3.13 -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Provide your group token (only needed for the first download)
cp .env.example .env            # then paste the token after MIAX_AE_TOKEN=
```

The data is served only by the workshop web-app (train and validation;
test never leaves the server). The first run downloads it with your token
and caches it in `data/raw/`. Later runs work offline. If there is no cache
and no token, the program stops with a `FileNotFoundError` explaining what
to do.

## Usage

```bash
python main.py
```

| Option | Description |
|---|---|
| `--model {denoising,baseline}` | `denoising` (default) is the corrected model. `baseline` reproduces the original 400-d ReLU submission. |
| `--latent-dim N` | Override the latent size of the chosen model. |
| `--refresh-data` | Re-download the data even if cached. |
| `--data-dir`, `--results-dir` | Cache and output locations (`data/raw/`, `results/`). |
| `--no-plots` | Skip figure rendering. |
| `--submit` | Submit to the **validation** leaderboard (repeatable). |
| `--submit --final --confirm-final` | **Irreversible** single test evaluation. It freezes the model. |
| `--log-level` | `DEBUG`, `INFO` (default), `WARNING`, `ERROR`. |

All hyper-parameters live in [`src/utils/config.py`](src/utils/config.py).
Run the tests with `pytest` and the style check with `flake8 src main.py tests`.

## Results

Measured locally with `python main.py` (seed 42, TensorFlow 2.21, CPU,
Windows). The PCA and equal-weight numbers match the reference notebook
exactly:

| Tracker (25 stocks unless noted) | Params | TE train | TE validation | Beats PCA |
|---|---|---|---|---|
| Equal-weight universe, 342 stocks (irreducible floor) | – | – | 2.16 % | – |
| PCA, 3 components (ghost competitor) | – | 4.40 % | 4.84 % | – |
| **Denoising AE, latent 8 (corrected, default)** | 228 k | **3.74 %** | **4.63 %** | **yes (−0.21 pp)** |
| Original submission: ReLU AE, latent 400 (`--model baseline`) | 1.41 M | 4.53 % | 4.99 % | no |

For reference, the professor's notebook reports a validation TE of 4.63 %
and a test TE of 4.35 %. The original submission scored 5.31 % on the
validation leaderboard (81st percentile against the monkeys).

The baseline reaches a **lower reconstruction loss** (0.808 vs 0.838) but
a **worse tracking error**. A 400-dimensional latent space memorises the
training data instead of extracting common factors. A good latent space
is not the one that reconstructs best.

Outputs written to `results/` (git-ignored):

- `metrics.json`: TE, correlation, CAGR, volatility and max drawdown for
  the AE, PCA and equal-weight universe, plus training summary.
- `tracker.csv`: the 25 tickers and their weights. `latent.csv`: the
  embedding of every stock.
- `latent_feature_correlation.csv`: correlation of each latent dimension
  with volatility, growth and beta.
- `figures/`:
  - `training_loss.png`: train/validation loss curves.
  - `latent_scatter.png`: first two latent dimensions.
  - `latent_tsne_by_sector.png`: sectors separate without the sector
    being an input.
  - `latent_feature_gradients.png`: latent space coloured by volatility,
    growth and beta.
  - `latent_feature_heatmap.png`: one latent dimension correlates 0.74
    with beta and 0.64 with volatility.
  - `validation_cumulative_returns.png`: AE, PCA and RSP on validation.

Validation is a model-selection tool, not a target. The test period
(a random one-year window in 2024–2025) is evaluated once and
irreversibly, so tuning many variants against validation overfits it.

## Architecture

See [docs/architecture.md](docs/architecture.md) for diagrams.

## References

- Professor's solution: `professor_solution/`
- Assignment materials: `materials/`
