# Architecture

## 1. Project overview

The project builds a **sparse index tracker**: a 25-stock portfolio that
replicates the S&P 500 Equal Weight index (ETF **RSP**) with the lowest
possible out-of-sample tracking error. The 25 stocks are not hand-picked:
an **autoencoder** compresses each stock's daily return history into a
low-dimensional embedding. A fixed procedure then groups the embeddings
(spherical k-means), takes one medoid per group and weights it by group
size. The only free component is the autoencoder. It has to beat a linear
PCA baseline (the leaderboard's "ghost competitor") and sector-stratified
random portfolios ("monkeys").

## 2. Module dependency diagram

```mermaid
graph TD
    MAIN[main.py<br/>CLI entry point] --> PIPE[src/pipeline.py<br/>orchestration]
    MAIN --> API[src/data/api_client.py]
    MAIN --> SUB[src/submission/client.py]
    MAIN --> LOG[src/utils/logging_config.py]
    MAIN --> CFG[src/utils/config.py]

    PIPE --> PRE[src/data/preprocessing.py]
    PIPE --> AE[src/models/autoencoder.py]
    PIPE --> TRN[src/models/training.py]
    PIPE --> PCA[src/models/pca_baseline.py]
    PIPE --> SEL[src/portfolio/selection.py<br/>FIXED contract]
    PIPE --> BT[src/portfolio/backtest.py]
    PIPE --> TRK[src/evaluation/tracker.py]
    PIPE --> XAI[src/evaluation/explainability.py]
    PIPE --> PLT[src/visualization/plots.py]
    PIPE --> SEED[src/utils/seed.py]

    API --> DS[src/data/dataset.py]
    API --> PRE
    PRE --> DS
    TRK --> BT
    TRK --> MET[src/evaluation/metrics.py]
    BT --> PRE
    XAI --> PRE

    AE --> CFG
    TRN --> CFG
    PCA --> CFG
    MET --> CFG
    PLT --> CFG
    SUB --> CFG
```

## 3. Data flow

```mermaid
flowchart LR
    A[(Web-app API<br/>train + validation)] -->|Bearer token| B[data/raw cache<br/>metadata.json<br/>prices.parquet<br/>benchmark.parquet]
    B --> C[Wide price panel<br/>date x ticker]
    C --> D[Simple returns<br/>TRAIN window only]
    D --> E[X: one row per stock<br/>~342 x ~1762]
    E --> F[StandardScaler<br/>per-day z-score<br/>fit on train]
    F --> G1[PCA, 3 components]
    F --> G2[Denoising autoencoder<br/>noise, tanh 64, latent 8]
    G1 --> H1[PCA latent]
    G2 --> H2[AE latent]
    H1 --> I[select_tracker<br/>unit sphere, k-means 25<br/>medoid, size weights]
    H2 --> I
    I --> J[25 tickers + weights]
    J --> K[Quarterly rebalanced<br/>backtest]
    C --> K
    K --> L[TE = sqrt 252 * std of r_track - r_RSP<br/>train / validation]
    H2 --> M[Explainability<br/>t-SNE by sector,<br/>vol / growth / beta]
    L --> N[(results/<br/>metrics.json, tracker.csv,<br/>latent.csv, figures)]
    M --> N
    J -. optional --submit .-> A
```

Only the **training** window reaches the autoencoder. Validation (2022–2023)
is used for model selection only. Test (a random one-year window in
2024–2025) never leaves the server and is evaluated once, irreversibly.

## 4. Class diagram

```mermaid
classDiagram
    class RunConfig {
        +str model
        +DenoisingAEConfig denoising
        +BaselineAEConfig baseline
        +TrainingConfig training
        +PlotConfig plots
        +int seed
        +int n_assets
        +bool make_plots
        +Path results_dir
        +latent_dim() int
    }
    class DenoisingAEConfig {
        +int latent_dim = 8
        +int hidden_units = 64
        +str hidden_activation = "tanh"
        +float noise_stddev = 0.3
        +float dropout_rate = 0.0
    }
    class BaselineAEConfig {
        +int latent_dim = 400
        +str latent_activation = "relu"
    }
    class TrainingConfig {
        +int epochs = 200
        +int batch_size = 32
        +float learning_rate
        +float validation_split
        +int early_stopping_patience
    }
    class DateWindow {
        +str name
        +Timestamp start
        +Timestamp end
        +from_metadata(name, spec) DateWindow
        +slice(obj)
    }
    class MarketDataset {
        +dict metadata
        +DataFrame prices_wide
        +Series benchmark
        +train_window() DateWindow
        +validation_window() DateWindow
        +n_assets() int
        +sectors() dict
    }
    class TrainingResult {
        +dict history
        +int epochs_run
        +int best_epoch
        +float best_val_loss
        +bool stopped_early
    }
    class PerformanceSummary {
        +float tracking_error
        +float correlation
        +float cagr_tracker
        +float vol_tracker
        +float maxdd_tracker
        +to_frame(name) DataFrame
    }
    class TrackerEvaluation {
        +str name
        +dict weights
        +Series returns
        +float te_train
        +float te_validation
        +PerformanceSummary validation_summary
    }
    class PipelineResult {
        +list tickers
        +ndarray latent
        +Model autoencoder
        +beats_pca() bool
        +comparison_frame() DataFrame
    }

    RunConfig *-- DenoisingAEConfig
    RunConfig *-- BaselineAEConfig
    RunConfig *-- TrainingConfig
    MarketDataset ..> DateWindow : creates
    TrackerEvaluation *-- PerformanceSummary
    PipelineResult *-- RunConfig
    PipelineResult *-- TrainingResult
    PipelineResult *-- "2" TrackerEvaluation : AE, PCA
    PipelineResult *-- PerformanceSummary : EW floor
```

## 5. Key design decisions

| Decision | Rationale |
|---|---|
| Denoising AE, latent 8, `tanh` 64, Gaussian noise 0.3 | Professor's reference. With about 342 samples and about 1762 features, a small regularised network generalises. A wide or deep one memorises. |
| Linear bottleneck | Signed coordinates and no dead ReLU units, so every stock has a direction for the cosine-based selection. |
| `StandardScaler` fitted on train only | Per-day cross-sectional z-score. Fitting on validation/test would be look-ahead leakage. |
| `selection.py` logic untouched | The server re-runs it to verify coherence. Any change gets the submission rejected. Its signature is covered by a unit test. |
| Seed 42 + TF op determinism | The server rejects other seeds. Determinism makes runs repeatable. |
| Local parquet cache | Offline, fast and reproducible reruns. Only the first run needs the token. |
| Token read from environment / `.env` | Keeps secrets out of the code and out of version control. |

## 6. How to run

```bash
python -m venv .venv                 # Python 3.11-3.13
source .venv/bin/activate            # Linux / macOS
.venv\Scripts\activate               # Windows
pip install -r requirements.txt
export MIAX_AE_TOKEN=<group token>   # first run only (or use .env)
python main.py                       # corrected denoising model
python main.py --model baseline      # original submission, for contrast
pytest                               # test suite (offline)
```
