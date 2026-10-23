# RAMSeS — Robust & Adaptive Model Selection for Time-Series Anomaly Detection

**RAMSeS** is a research framework for **unsupervised time-series anomaly detection (TSAD)** that automatically selects and deploys the best detection strategy from two complementary approaches:

1. **Optimized Ensemble** — A stacking ensemble discovered by a Genetic Algorithm (GA) with configurable meta-learners (RF/LR/GBM/SVM)
2. **Robust Single Model** — Selected via Linear Thompson Sampling with comprehensive robustness testing

The framework provides an end-to-end pipeline for data loading, model training, robustness evaluation, and adaptive online selection.

> **Research Group:** D2IP @ TU Berlin  
> **License:** Apache 2.0  
> **Status:** Research code (actively maintained)

---

## ✨ Key Features

### Dual Selection Strategy
- **Ensemble Branch:** Genetic Algorithm optimizes detector combinations; meta-learner stacks anomaly scores
- **Single-Model Branch:** Multi-criteria evaluation combining:
  - Linear Thompson Sampling with ε-greedy exploration
  - GAN-based robustness testing (borderline synthetic anomalies)
  - Off-by-threshold sensitivity analysis
  - Monte Carlo noise stress tests
  - Markov-chain rank aggregation

### Online Adaptation
- Sliding window processing of the last 20% of the series
- Four re-optimization scenarios: fixed, periodic, relative threshold and
  adaptive threshold, comparable side by side on one replay of the same windows
- Background re-optimization: the stream never waits for it
- Optional regime shift injection for testing distribution changes

### Comprehensive Evaluation
- Automatic comparison of ensemble vs. single-model performance
- Detailed performance tracking (F1, PR-AUC, fitness)
- Memory and computational overhead monitoring


---

## 📁 Repository Structure

```text
RAMSeS/
├── app.py                              # Main pipeline entrypoint
├── run_full_testbed.py                 # Batch runner over a testbed file list
├── run_testbed_comprehensive.py        # Batch runner + comprehensive reporting
├── requirements.txt                    # Python dependencies (requirements-server.txt for deployment)
├── environment.yml                     # Conda environment (environment.full.yml pins transitives)
├── LICENSE                             # Apache 2.0 license
│
├── Algorithms/                         # Base anomaly detector implementations
│   ├── base_model.py                   # PyMADModel interface every detector implements
│   ├── pyod_model.py                   # Adapter for PyOD estimators
│   ├── abod.py                         # Angle-Based Outlier Detection
│   ├── alad.py                         # Adversarial Learned Anomaly Detection
│   ├── anomaly_transformer.py          # Transformer-based detector
│   ├── cblof.py                        # Cluster-Based Local Outlier Factor
│   ├── cof.py                          # Connectivity-Based Outlier Factor
│   ├── dghl.py                         # Deep Generative Hierarchical Learning
│   ├── kde.py                          # Kernel Density Estimation
│   ├── lof.py                          # Local Outlier Factor
│   ├── lstmvae.py                      # LSTM Variational Autoencoder
│   ├── mean_deviation.py               # Mean Deviation
│   ├── mtad_gat.py                     # Multivariate TSAD via Graph Attention
│   ├── nearest_neighbors.py            # k-Nearest Neighbors
│   ├── rnn.py                          # Recurrent Neural Network
│   ├── running_mean.py                 # Running Mean
│   ├── sos.py                          # Stochastic Outlier Selection
│   ├── chronos_detector.py             # Chronos foundation-model detector
│   ├── series2graph_detector.py        # Series2Graph (fetched, not vendored — see below)
│   ├── windowed.py                     # Windowing wrapper shared by the detectors
│   └── tsb_ad/                         # Vendored TSB-AD detector suite (Apache 2.0)
│
├── Configs/                            # Configuration files
│   ├── config.yml                      # Default configuration (read by Utils and the WebUI)
│   └── custom_config.yml               # Testbed configuration (run_full_testbed.py default)
│
├── Datasets/                           # Data loading utilities
│   ├── dataset.py                      # Dataset class definitions
│   └── load.py                         # Data loader implementations
├── Loaders/                            # Entity/tensor loaders feeding the trainers
│
├── Model_Training/                     # Model training and management
│   ├── train.py                        # Training orchestration
│   ├── trainer.py                      # Training logic
│   ├── entities.py                     # Entity/dataset record types
│   └── hyperparameter_grids.py         # Hyperparameter configurations
├── Model_Optimization/                 # Hyperparameter search over the grids
│
├── Model_Selection/                    # Model selection algorithms
│   ├── Thompson_Sampling.py            # Linear Thompson Sampling + sliding windows
│   ├── rank_aggregation.py             # Markov-chain rank aggregation
│   ├── inject_anomalies.py             # Synthetic anomaly injection
│   ├── anomaly_parameters.py           # Injection parameter definitions
│   └── Sensitivity_robustness/
│       ├── GAN_test.py                 # GAN-based robustness testing
│       ├── Monte_Carlo_Simulation.py   # Monte Carlo noise stress tests
│       └── off_by_threshold_testing.py # Borderline sensitivity analysis
│
├── Metrics/                            # Evaluation and GA components
│   ├── Ensemble_GA.py                  # Genetic Algorithm for ensemble optimization
│   ├── Ensemble_Genetics.py            # GA operators and fitness plumbing
│   ├── metrics.py                      # Performance metrics (F1, PR-AUC, etc.)
│   └── ranking_metrics.py              # Ranking evaluation utilities
├── distributions/                      # Mallows / Plackett-Luce rank distributions
├── vus/                                # Volume-Under-Surface metrics + robustness eval
│
├── Explainability/                     # Grounded natural-language explanations
│   ├── ir.py                           # Intermediate Representation: facts as JSON "atoms"
│   ├── llm.py                          # LLM client
│   ├── narrate.py                      # Renders the IR into prose
│   └── verifier.py                     # Scores narratives for hallucination/omission
│                                       #   against the IR they were generated from
│
├── WebUI/                              # Local Flask UI: configure, watch, explain
│   ├── server.py                       # Routes, JSON API, SSE progress stream
│   ├── jobs.py                         # Drives app.py as a subprocess
│   ├── artifacts.py                    # Reads pipeline output off disk
│   └── static/, templates/             # Front-end assets
│                                       # Run with: python -m WebUI  (localhost only)
│
├── Controller/, Services/, dao/        # Layered access to stored runs (mdata/mmodel/mevaluation)
├── db/tsad.db                          # SQLite store backing the above
│
├── Utils/                              # Utility functions
│   ├── utils.py                        # CLI argument parsing, misc utilities
│   ├── config.py                       # Configuration file parser
│   ├── plotting.py                     # Visualization utilities
│   ├── pipeline_spec.py                # Detector/family definitions shared by UI and pipeline
│   └── results_formatter.py            # Results formatting and export
│
├── testbed/file_list/                  # CSV files listing datasets for batch runs
│   ├── test_single.csv                 # Single dataset test
│   ├── test_m_skab.csv                 # SKAB multivariate datasets
│   └── test_u_ucr_anomaly_archive.csv  # UCR univariate datasets
│
└── Misc/                               # One-off analysis and plotting scripts
```

### Directories that are not in version control

These are generated, fetched, or working data. All are gitignored; none are
needed to read or run the code, and several are large.

| Path | |
|---|---|
| `Mononito/` | **Symlink** to the dataset/checkpoint archive (~61 GB). See [Datasets](#-datasets). |
| `myresults/`, `results/`, `output/`, `Outputs/`, `testbed_results/` | Pipeline output, written per run |
| `saved_models/` | Checkpoints for one SMD entity; nothing in the codebase loads them |
| `logs/`, `dataset_lists/` | Run logs and generated dataset manifests |
| `docs/` | Working notes — `docs/guides/` is current, `docs/archive/` is historical. See `docs/README.md`. |
| `archive/` | Retired result trees, plots and code kept for reference |
| `demo_paper_prep/` | Demo-paper drafts and reference PDFs |

`Algorithms/tsb_ad/models/Series2Graph.py` is fetched rather than committed —
it is patent-encumbered and licensed for research use only, unlike the
Apache-2.0 TSB-AD code vendored around it:

```bash
python -m Algorithms.tsb_ad.fetch_series2graph
```

---




## 📦 Datasets

RAMSeS uses the **Mononito** time-series repository ([arXiv:2210.01078](https://arxiv.org/abs/2210.01078)), which includes:

- **SKAB** — Multivariate sensor data from industrial systems
- **SMD** — Server Machine Dataset (multivariate)
- **UCR Anomaly Archive** — Univariate time series

### Download Instructions

1. **Download the Mononito dataset** from Google Drive:  
   [https://drive.google.com/drive/folders/1BLcaGm4bNSBueh3Hy_-dP1MKNhzfulwC?usp=share_link](https://drive.google.com/drive/folders/1BLcaGm4bNSBueh3Hy_-dP1MKNhzfulwC?usp=share_link)

2. **Extract and organize:**
   ```bash
   # Example structure
   ~/Mononito/
   ├── datasets/
   │   ├── SKAB/
   │   ├── SMD/
   │   └── UCR/
   └── trained_models/  # Created during training
   ```

3. **Symlink it into the repo** (recommended) so the relative paths baked into
   the code — e.g. `Model_Training/train.py`'s `save_dir='Mononito/trained_models'`
   — keep working, and the bulk data stays out of the repo tree:
   ```bash
   ln -s /path/to/Mononito Mononito   # Mononito/ is gitignored
   ```

4. **Update configuration** in `Configs/config.yml`. These values are used
   verbatim (they are *not* resolved against the repo root), so use absolute
   paths. Pointing them at the symlink means a future data move only requires
   re-pointing the symlink:
   ```yaml
   dataset_path: "/path/to/RAMSeS/Mononito/datasets"
   trained_model_path: "/path/to/RAMSeS/Mononito/trained_models"
   ```

> **Licensing Note:** Please follow the original dataset licenses and cite the appropriate papers when using these datasets.

### TSB-AD (real labels)

[TSB-AD](https://thedatumorg.github.io/TSB-AD/) (Liu & Paparrizos, *The Elephant
in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark*,
NeurIPS 2024 Datasets & Benchmarks) adds two collections with **real anomaly
labels**:

- **TSB-AD-U** — 870 univariate series (dataset key `tsb-ad-u`)
- **TSB-AD-M** — 200 multivariate series, 2 to 248 channels (dataset key `tsb-ad-m`)

Where it lives: the downloaded CSVs are kept outside the repository, at
`ramses-data/TSB-AD/TSB-AD-U/*.csv` and `ramses-data/TSB-AD/TSB-AD-M/*.csv`, and
symlinked into the data root (`Datasets/TSB-AD-U`, `Datasets/TSB-AD-M`, which
are untracked like the other dataset links). Dataset names are case-insensitive
(`--dataset TSB-AD-M` works).

Each file is one series, named
`[index]_[Dataset]_id_[id]_[Domain]_tr_[TrainEnd]_1st_[FirstAnomaly].csv`, with one
column per channel plus `Label`. The loader (`Datasets/load.py`, `load_tsb_ad`;
file names in `Datasets/tsb_ad.py`) takes samples `[0, TrainEnd)` as the training
split and the rest as the test split with its real labels, scales each channel
with the training split's range, and downsamples as for UCR (block maxima,
labels included; skipped when either split would fall below `min_length`).

**Entity ids** are the file name up to `_tr_`, e.g. `001_NAB_id_1_Facility`
(`--entity 001_NAB_id_1_Facility`). The full stem, the file name and the bare
index (`1`, `001`) are accepted and resolve to that id, so checkpoints and
results always land in one directory per series. The web UI lists every series
under its id; type to filter the 870.

Caveats measured on the download: the training split is not always anomaly-free
(177 TSB-AD-U and 22 TSB-AD-M series label samples before `TrainEnd`, mostly
YAHOO, Stock and TAO); the split is taken from the file name all the same, as
TSB-AD does, and detectors train on it unsupervised. 494 of the 870 TSB-AD-U
series (57%) and 80 of the 200 TSB-AD-M series (40%) have no labelled anomaly in
their online part — the last 20% of the test split — and so cannot run the
online phase on real labels (see `--online_labels`). Univariate-only families (POLY, TimesFM, Series2Graph) are
dropped on TSB-AD-M and multivariate-only ones (ABOD) on TSB-AD-U, as for every
dataset.

> **Licence:** the TSB-AD preprocessing and its code are Apache-2.0; every
> series keeps the licence of its original source dataset (NAB, YAHOO, MSL,
> SMAP, SMD, UCR, …), listed per dataset on the
> [TSB-AD page](https://thedatumorg.github.io/TSB-AD/). Cite TSB-AD and the
> original source when you use them.

---


## ⚙️ Installation

### Prerequisites
- Python 3.9+ (tested with Python 3.11)
- CUDA-capable GPU (optional, CPU mode supported)
- 8GB+ RAM recommended

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/Maxoud99/RAMSeS.git
cd RAMSeS

# Create and activate conda environment
conda env create -f environment.yml
conda activate RAMS

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Option 2: Pip + Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Maxoud99/RAMSeS.git
cd RAMSeS

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### Key Dependencies
- **PyTorch** 2.5 (with optional CUDA support)
- **TensorFlow** 2.18 (GAN robustness stage only; runs on the CPU, see below)
- **scikit-learn** 1.7 (ML algorithms)
- **PyOD** 3.6 (outlier detection)
- **NumPy**, **Pandas**, **Matplotlib** (data processing & visualization)
- **loguru** (structured logging)
- **Flask** 3.1 (serves the local WebUI)

All 26 dependencies are pinned in `requirements.txt`; `environment.yml`
installs from it, so the conda and pip routes cannot drift apart.

### On a machine without a GPU

Use `requirements-server.txt` instead. It swaps `tensorflow` for
`tensorflow-cpu` (230 MB against 616 MB; TensorFlow is used by the GAN
robustness stage alone) and takes torch from PyTorch's CPU index, avoiding
~5 GB of `nvidia-*` CUDA runtime that a GPU-less box cannot use. Its header
documents the reasoning and the exact two-command install.

Do not install both files into one environment: `requirements.txt` brings
`tensorflow` and `requirements-server.txt` brings `tensorflow-cpu`, and the two
distributions overwrite each other's `tensorflow` package.

### TensorFlow and the GPU

On a GPU machine, TensorFlow is kept on the CPU and torch keeps the GPU.
TensorFlow is used only by the GAN robustness stage, and the pinned versions
cannot share a GPU in one process:

- TensorFlow 2.18.1 is built against CUDA 12.5 and cuDNN 9.3.
- torch 2.5.1 ships cuDNN 9.1.0. The conda build in `environment.full.yml`
  (`cuda11.8_cudnn9.1.0`) bundles it in `torch/lib`, and the PyPI wheel pins
  `nvidia-cudnn-cu12==9.1.0.70`.

`app.py` loads torch first, so its cuDNN 9.1.0 is already in the process when
TensorFlow looks for `libcudnn.so.9`. TensorFlow needs at least the minor
version it was built with, so it refuses to start cuDNN on the GPU
(`Loaded runtime CuDNN library: 9.1.0 but source was compiled with: 9.3.0`),
and the GAN stage fails. `tensorflow-cpu` (`requirements-server.txt`) never
sees a GPU and is not affected.
`pip install tensorflow[and-cuda]` does not fix this: it pins
`nvidia-cudnn-cu12==9.3.0.75`, which conflicts with torch's pin.

`Utils/tf_device.py` hides the GPUs from TensorFlow only, through
`tf.config.set_visible_devices([], 'GPU')`. `CUDA_VISIBLE_DEVICES` is not used
because it would hide the GPU from the detectors too. The GAN networks are
small, so the CPU costs little time. If your TensorFlow CUDA stack does match,
for example a torch built against cuDNN 9.3 or later, opt back in with:

```bash
export RAMSES_TF_GPU=1
```

The GAN stage also fails soft. If it raises for any reason, the run logs one
warning ("GAN robustness test failed and was skipped"), which the web UI shows
on the run. The run then continues as with `--skip_gan`: Off-by-threshold,
Monte Carlo and rank aggregation run without the GAN ranking.

---

## 🚀 Quick Start

### 1. Configure Paths

Edit `Configs/config.yml` to set your dataset and model paths:

```yaml
# Paths
dataset_path: "/path/to/Mononito/datasets"
trained_model_path: "/path/to/Mononito/trained_models"

# Training parameters
downsampling: 10
min_length: 256
training_size: 1.0
model_architectures: 'all'  # or specify: 'LOF,CBLOF,NN'
training_timeout: 3600      # seconds per detector instance; null or 0 = no limit

# Evaluation parameters
normalize: True
evaluation_metric: 'Best F-1'
```

### 2. Train Base Models

RAMSeS automatically trains missing models on first run:

```bash
python app.py \
  -c Configs/config.yml \
  --dataset SKAB \
  --entity 5
```

This will:
- Load training data from the specified dataset/entity
- Train any missing base detectors (LOF, CBLOF, NN, etc.)
- Save trained models to `trained_model_path/SKAB/5/`

### 3. Run Model Selection

Execute the complete RAMSeS pipeline:

```bash
python app.py \
  -c Configs/config.yml \
  --dataset SKAB \
  --entity 5 \
  --parallel false
```

**Pipeline stages:**
1. Load training and test data
2. Train/load base detector models
3. Inject synthetic anomalies
4. Run Genetic Algorithm for ensemble optimization
5. Execute Thompson Sampling for single model selection
6. Perform robustness tests (GAN, Monte Carlo, Off-by-Threshold)
7. Aggregate rankings and select final model

### 4. View Results

Results are saved in human-readable format with comprehensive metrics and visualizations.

**Example output:**
```
FRAMEWORK FINAL DECISION
================================================================================

Single Model Option:
--------------------------------------------------
  Model       : LOF_2
  F1 Score    : 0.8542
  PR-AUC      : 0.7891

Ensemble Option:
--------------------------------------------------
  Models      : ['LOF_1', 'CBLOF_3', 'NN_2']
  Size        : 3
  Meta-Model  : rf
  F1 Score    : 0.8721
  PR-AUC      : 0.8103
  Fitness     : 0.8412

Final Choice:
--------------------------------------------------
  ✓ ENSEMBLE SELECTED
    Reason: Ensemble F1 (0.8721) >= Single Model F1 (0.8542)
    Improvement: +0.0179 (2.10%)
```

---

## 🖥️ Web UI

An optional local UI for configuring a run, watching it live, and reading the
explanation it produces:

```bash
python -m WebUI              # http://127.0.0.1:5000
python -m WebUI --port 8080  # different port
```

It drives `app.py` as a subprocess and reads the artifacts off disk, so it never
imports the pipeline, torch or matplotlib — the server starts instantly and does
not hold the pipeline's working set. Flask and the standard library are its only
dependencies.

> **Bound to localhost by design.** There is no authentication, and the app
> spawns processes with user-supplied arguments. Do not bind it to a public
> interface.

---

## 🎯 Configuration Options

### Command-Line Arguments

```bash
python app.py \
  -c, --config_file_path <path>   # Config file (default: Configs/config.yml)
  --dataset <name>                # Dataset name (e.g. SKAB, SMD, UCR, tsb-ad-u, tsb-ad-m)
  --entity <id>                   # Entity ID within the dataset
  --detectors <list>              # Comma-separated detectors to select among,
                                  #   e.g. 'LOF_1,NN_2,CBLOF_3'. Only the families
                                  #   of the named detectors are trained. Untrained
                                  #   ones are skipped with a warning; at least two
                                  #   must remain. Default: all of them.
  --stages <list>                 # Sub-stages to run: ga, thompson, gan, offby,
                                  #   montecarlo, plus 'robustness' (= gan,offby,
                                  #   montecarlo) and 'all' (default). Any strict
                                  #   subset stops after those stages — no rank
                                  #   aggregation, final decision or online phase —
                                  #   and runs sequentially.
  --overwrite <true|false>        # Retrain base detectors even when checkpoints
                                  #   exist; overrides `overwrite` in the config.
                                  #   Training dominates runtime, so 'false' is
                                  #   much faster.
  --anomaly_type <name>           # Synthetic anomaly injected at stage 4: spikes
                                  #   (default), contextual, flip, speedup, noise,
                                  #   cutoff, scale, wander, average
  --anomaly_rate <float>          # Target fraction of timesteps labelled anomalous,
                                  #   in (0, 1]. For 'spikes' this is the per-timestep
                                  #   injection probability; for other types it sizes
                                  #   the injected segment. Omit for per-type defaults.
  --decision_metric <list>        # Metrics the fitness function is built from: f1,
                                  #   pr_auc (both by default), vus. Each may carry a
                                  #   weight ('f1:0.5,pr_auc:0.3,vus:0.2'); unweighted
                                  #   metrics count equally and weights are normalised.
                                  #   The weighted mean is what the GA, Thompson
                                  #   Sampling and the final ensemble-vs-single
                                  #   comparison all maximise.
  --parallel <true|false>         # Enable parallel model selection
  --skip_gan                      # Skip GAN robustness testing (faster; debugging)
  --explain                       # Write explainability reports and plots.
                                  #   OFF by default.
  --llm_model <name>              # Model for LLM narration after an --explain run
                                  #   (default: qwen2.5:14b-instruct)
  --llm_base_url <url>            # OpenAI-compatible endpoint for narration
                                  #   (default: http://localhost:11434/v1, i.e. Ollama).
                                  #   Narration is skipped with a warning if no
                                  #   server is reachable.
```

> Use `--config_file_path` (or `-c`), not `--config`. The short form works today
> only because argparse accepts unambiguous prefixes, and it would break the
> moment a second `--config*` option is added.

### Online Mode (Adaptive Selection)

Enable online learning and adaptive model selection:

```bash
python app.py \
  -c Configs/config.yml \
  --dataset SMD \
  --entity machine-1-1 \
  --enable_online \
  --strategy all \
  --update_interval 5 --drop_pct 20 --adaptive_window 10 --adaptive_k 2 \
  --reopt_delay 2 --reopt_timeout 3600 \
  --iteration 5 \
  --max_online_windows 100 \
  --overwrite false
```

The last 20% of the test series is the online part. What its windows are scored
on is the run's **label source** (`--online_labels`, below): the series' own
labels where the dataset has them, synthetic anomalies otherwise. With injected
anomalies, they are injected **once**, seeded, into the whole online part, and every window is cut
from that one stream, so a sample has the same value and label in every
(95%-overlapping) window that contains it. Window 0, the first online window, is
replayed. For an anomaly type that injects several copies of the series
(`speedup`, `scale` and `wander` make 4, `cutoff` and `average` 2), every window
holds the **same time slice of each copy**, concatenated in copy order, as the
offline phase concatenates the copies; the number of windows is the plain online
part's. `speedup` resamples its anomalous segment, so its copies differ in
length; a window then takes from each copy the samples covering the same
original times, and its `segments` record where each copy's part lies.

On every window the deployed GA ensemble's fitness (through the GA's own fitted
meta-model) and the deployed single model's score are computed with the run's
`--decision_metric`. When a scenario triggers, a re-optimization re-runs model
selection (GAN skipped, explanations off) in a **separate process**, seeded with
the run's seed plus the triggering window, on the offline phase's injected data
slid forward by the injected online samples seen so far (each sample once), with
the robustness stages on the same window before injection, as offline. Its
result is switched in at exactly window *trigger + L* (`--reopt_delay`); if it is
not ready there, the stream waits for it, and the wait is recorded. Triggers from
the trigger up to the switch are suppressed (recorded, not queued). A
re-optimization that would be switched in after the last window is killed when
the stream ends and recorded as not applied. One that fails or exceeds
`--reopt_timeout` is recorded as failed and the current selection stays. A second
scenario that triggers at the same window reuses the first one's result, which
is identical by construction.

**Online mode parameters** (each also a key in `Configs/config.yml`, which a
flag overrides):
- `--enable_online` — Enable online phase processing
- `--strategy <scenario>` (config `online_strategy`, default `periodic`):
  - `fixed` — keep the offline selection for the whole stream
  - `periodic` — re-optimize at windows M, 2M, … (`--update_interval`, default 5)
  - `threshold` — re-optimize when the ensemble's or the single model's score
    falls more than X% (`--drop_pct`, default 20) below its baseline, the first
    score after the last (re)optimization (window 0 for the offline selection)
  - `adaptive_threshold` — re-optimize when a score falls below mean − k·std
    (`--adaptive_k`, default 2) of its last N scores (`--adaptive_window`,
    default 10); the history restarts when a re-optimization is switched in and
    the X% rule applies until N scores exist
  - `all` — run the offline phase once, then all four scenarios on the same
    windows, each from the same offline selection
  - `fixed-random` — a random selection kept throughout (command-line baseline)
  - Aliases: `adaptive` = `periodic`, `fixed-best` = `fixed`
- `--reopt_delay <L>` (config `reopt_delay`, default 2, at least 0) — windows
  between a trigger and the switch to its result. With 0 the triggering window
  itself is scored again with the new selection.
- `--reopt_timeout <seconds>` (config `reopt_timeout`, default 3600; 0 or `null`
  for no limit) — how long one re-optimization may run before it is killed.
- `--iteration <n>` — Number of iterations for window sizing (default: 5).
  Windows are at least 100 samples long, so an entity needs at least 100 online
  samples (a test series of 500 or more) for the phase to run. SKAB's test
  series (about 230 samples) are too short; the run then logs
  `ONLINE PHASE SKIPPED` and says why, and the web UI refuses `enable_online`
  for such an entity with the same reason
  (`GET /api/online/feasibility/<dataset>/<entity>?iteration=I`).
- `--inject_online_regime` — Inject regime shifts (scale + wander) into the
  online part: the online stream becomes 8 shifted copies of it, one after the
  other, and window positions (`start`/`end`, `online_start`/`online_end`) count
  in that stream (`summary.window_coordinates` is then `online_stream`), so they
  run past the end of the test series. Whether the phase can run is still judged
  on the plain online part.
- `--max_online_windows <n>` — Limit number of online windows processed
- `--online_labels <real|injected>` (config `online_labels`, default: `real`
  wherever the dataset provides labelled test anomalies, `injected` otherwise;
  `auto` on the command line restores that default over the config) — what the
  online windows are scored on. See *Real labels online* below.

#### Real labels online (`--online_labels real`)

Evaluating online adaptation on injected anomalies measures how well a
selection finds what was injected. Where the test series carries **real**
labels, the online phase uses them instead:

- **Nothing is injected online.** Every window is the raw test slice with its
  true labels (`summary.injection` is `null`, `summary.label_source` is
  `"real"`), so a window's data is exactly the test series between its
  `start` and `end`.
- **Both window signals** (the deployed ensemble's fitness and the single
  model's score, on the run's decision metric) are computed on the true
  labels. A window whose true labels are one class — most windows of most
  series, since anomalies are rare — has **no score** (`null`): it is drawn as
  a gap, cannot trigger, and does not enter the adaptive history. Each window
  records `n_anomalous` (true) and `scorable`; the summary records
  `n_windows_scorable` and `n_windows_with_anomaly`.
- **Threshold and adaptive-threshold triggers** react to those true-label
  scores, and the periodic scenario re-optimizes on its schedule as before.
- **A re-optimization is a supervised re-selection on the labels seen so
  far:** the offline part's raw samples slid forward by the online samples seen
  up to the triggering window, each sample once, with their real labels, and
  no injection anywhere (it assumes the labels are fed back as the stream
  goes). Every stage it runs (GA, Thompson Sampling, off-by-threshold, Monte
  Carlo; GAN is skipped) takes those labels. Thompson Sampling skips windows
  whose labels are one class instead of rewarding a detector for an undefined
  metric. One whose data holds no labelled anomaly at all has nothing to select
  on: it is refused before it starts and recorded as a failed
  re-optimization, and the current selection stays.
- **The offline phase is unchanged**: it still selects on injected anomalies.
- **Fairness is kept**: all four scenarios replay identical windows, and the
  scores are the same seeded, memoised ones in every scenario.

Datasets whose loader provides real test labels: TSB-AD-U, TSB-AD-M, UCR
(`anomaly_archive`), SMD, MSL, SMAP, and any dataset in the
`train/ test/ test_label/` layout. **SKAB does not**: its loader labels with the
CSV's last column, which is `changepoint`, not `anomaly`, so SKAB defaults to
injected and refuses `--online_labels real`. `--inject_online_regime` injects
into the online stream, so it needs `--online_labels injected`; combined with
real labels (also when they are only the default) the run stops with a message
before it trains anything.

An entity whose online part has **no** labelled anomaly has nothing to score
with real labels: the feasibility check
(`GET /api/online/feasibility/<dataset>/<entity>?iteration=I&online_labels=real`)
answers `ok: false` with the reason "No labelled anomaly in the online part —
nothing to score", the web UI refuses `enable_online` for it, and a run logs
`ONLINE PHASE SKIPPED`. The check also reports how many online windows contain
a labelled anomaly (`n_windows_with_anomaly`) and how many can be scored
(`n_windows_scorable`). With the default `--iteration 5` windows, 21% of the
online windows of TSB-AD-U and 35% of TSB-AD-M contain one, counted over every
series.
`--online_labels injected` runs such an entity on injected anomalies, as before.

> **Offline results may shift.** The offline phase's robustness tests (GAN,
> off-by-threshold, Monte Carlo) now score only the offline 80% of the test
> series. They used to receive the whole series, the online 20% included, which
> leaked future data into the offline selection. Rankings, and so the offline
> selection and its scores, can differ from results produced before this change.

Results are written, and rewritten after every window (and every 15 s while
the stream waits for a re-optimization), to
`results/online/<dataset>/<entity>/` — `summary.json` plus one `<scenario>.json`
per scenario. The schema is documented in `Model_Selection/online_phase.py`:
among other things it records the delay L and timeout, every wait
(`waiting` events, `waited_s`, the scenario status `waiting_for_reopt` with the
windows it waits between), each re-optimization's trigger and switch-in window
(`apply_window`), the ones not applied because the stream ended, and the
feasibility verdict. The web UI serves both files at
`GET /api/online/<dataset>/<entity>`, and reports a run whose process died
mid-phase as `interrupted`.

**Live monitor.** The online page opens with a monitor of the stream itself:
the test series as the online phase replays it (offline part for context,
then the online part — injected values when injecting), its real anomalies as
solid bands, injected ones hatched and named by type, the offline/online
boundary, and the window being scored as a rectangle that moves while the run
is live. Under it, the chosen scenario's score track for that window: the
deployed single model's or ensemble's anomaly scores, the threshold they were
cut at, the flagged samples, and a mark wherever a re-optimization was
switched in. A finished run plays back (play, pause, step, scrub, speed), with
the view following the window or showing the whole stream; a channel picker
appears for multivariate series, and "Show data" lists the current window
sample by sample. The pipeline writes the data as it goes, beside
`summary.json` (`stream.json`, `stream.npz`, `stream/<scenario>/w<i>.npz`;
schema in `Model_Selection/online_stream.py`), and the page reads it from
`GET /api/online/<dataset>/<entity>/stream?channels=0,1&from=&to=&max_points=&copy=&scenario=&window=`,
which returns min/max-decimated slices (a one-sample spike survives any zoom)
and anomalies as exact spans (none is decimated away).

---

## 🧩 Available Algorithms

RAMSeS ships **34 detector families**, expanded into **107 configured detector
instances** (`LOF_1`, `LOF_2`, … — the same algorithm at different
hyperparameters, which is what gives the ensemble something to combine). Both
lists are defined once, in `Utils/pipeline_spec.py`; `DETECTOR_FAMILIES` and
`ALL_DETECTORS` there are the authority if this table drifts.

**Proximity and density** — `LOF`, `NN` (k-Nearest Neighbors), `CBLOF`, `COF`,
`ABOD`, `KDE`, `SOS`, `LUNAR`

**Linear, statistical and classical** — `PCA`, `OCSVM`, `MCD`, `IFOREST`,
`HBOS`, `MD` (Mean Deviation), `RM` (Running Mean), `SpectralResidual`,
`POLY` (univariate only), `KMEANSAD`

**Deep reconstruction and forecasting** — `AutoEncoder`, `RNN`, `LSTMVAE`,
`LSTMAD`, `DGHL`, `DONUT`, `OmniAnomaly`, `USAD`, `TRANAD`, `FITS`

**Transformer and foundation models** — `TIMESNET`, `OFA`, `TIMESFM`, `CHRONOS`

**Graph-based** — `Series2Graph`, `MTADGAT`

Training is per **family**, not per instance: selecting one untrained detector
trains its whole hyperparameter grid (`Utils.pipeline_spec.families_for`). That
is the difference between a short run and a long one.

Fourteen families run through TSB-AD's whole-series interface rather than PyOD
(`TSBAD_FAMILIES`); most are vendored under `Algorithms/tsb_ad/`, but `CHRONOS`
and `MTADGAT` live in `Algorithms/`, and `Series2Graph` must be fetched
separately because it is patent-encumbered:

```bash
python -m Algorithms.tsb_ad.fetch_series2graph
```

**Choosing a subset:**
```yaml
# Configs/config.yml
model_architectures: 'all'            # every family
model_architectures: 'LOF,CBLOF,NN'   # or a comma-separated subset
```
or per run, without editing the config:
```bash
python app.py --detectors LOF_1,CBLOF_3,NN_2 --dataset SKAB --entity 5
```

---

## 🧪 Running the Tests

```bash
python -m pip install pytest     # test-only; not in requirements.txt
./run_tests.sh                   # everything
./run_tests.sh Utils WebUI       # only these paths
```

Three things the script handles that a bare `pytest` invocation does not:

**One process per test module.** `Model_Selection/test_thompson_sampling.py`
and `Model_Selection/test_rank_aggregation.py` install fake `Metrics`, `Metrics.Ensemble_GA`
and `Metrics.metrics` entries into `sys.modules` at import time, so they can
exercise the module under test without pulling in the pipeline. Any test
sharing that interpreter afterwards receives the stubs instead of the real
package — running the suite in a single pytest process makes
`test_reward_domain` fail with "Metrics is not a package".

**`PYTHONPATH` = repo root plus each file's own directory.** The suite mixes
dotted imports (`from Metrics.metrics import ...`) with bare ones
(`from Thompson_Sampling import ...`), so neither path alone satisfies it.

**`MPLBACKEND=Agg`.** On a headless machine matplotlib's Qt backend aborts with
"Could not find the Qt platform plugin xcb".

Expected: **27 modules, ~980 tests and subtests, all passing.**

If you see a wall of `ModuleNotFoundError` for `torchinfo`, `arch`, `timesfm`
or `dill`, or `Invalid model name: SpectralResidual`, the environment has
drifted from `requirements.txt` rather than the code being broken — reinstall
it (see [Installation](#-installation)).

---

## 🧪 Reproducibility Tips

- **Fix random seeds** for all stochastic components (GA, Monte Carlo, GAN, Thompson Sampling).
  `app.py` applies `random_seed` from the config to Python's, NumPy's and torch's
  RNGs before training and again before stage 4, so a rerun starts selection from
  the same state whether its models were trained in-process, trained under
  `training_timeout` in child processes, or loaded from disk
- **Log hyperparameters** along with results
- **Use absolute paths** for `dataset_path` and `trained_model_path`
- **Version control** your configuration files
- **Document** dataset versions and preprocessing steps

---

## ❓ Troubleshooting

### Model Loading Issues

**Problem:** `Model X not found in save_dir`

**Solution:** 
- Ensure trained models exist at `trained_model_path/DATASET/ENTITY/`
- Check that model files are named correctly (e.g., `LOF_1.pth`, `CBLOF_2.pth`)
- Set `overwrite: True` in config to retrain models

**Problem:** `⚠ X failed to train and is left out of this run: ...`

**What happened:** training is per detector instance. One that raises, or
that runs past `training_timeout` (each instance trains in a child process
that is killed at the limit), is logged with its reason and left out; the run
continues as long as at least two detectors remain. After one instance of a
family times out, the rest of that family is not attempted (with
`--overwrite false`, instances already on disk are still used). SOS also refuses,
before fitting, any entity longer than 10,000 rows: it holds four n x n
matrices at once (~26 GB on a full-length SMD machine).

**Solution:**
- The reason is in the warning; the traceback is just above it in the log
- For SOS, or a timeout on a long entity, set `downsampling` in the config
- Raise `training_timeout` if a detector legitimately needs longer

### Dataset Loading Issues

**Problem:** Empty entities in train/test data

**Solution:**
- Verify `dataset_path` points to the correct Mononito root directory
- Check that dataset follows the expected structure in `Datasets/load.py`
- Ensure dataset name matches exactly (case-sensitive)

### Visualization Issues

**Problem:** Matplotlib display errors on remote servers

**Solution:**
```bash
# Set non-interactive backend before running
export MPLBACKEND=Agg
python app.py -c Configs/config.yml --dataset SKAB --entity 5
```

Or add to your Python script:
```python
import matplotlib
matplotlib.use('Agg')
```

---

## 📚 Citation

If you use RAMSeS in your research, please cite our paper:

```bibtex
@misc{abdelmaksoud2026ramsesrobustadaptivemodel,
      title={RAMSeS: Robust and Adaptive Model Selection for Time-Series Anomaly Detection Algorithms}, 
      author={Mohamed Abdelmaksoud and Sheng Ding and Andrey Morozov and Ziawasch Abedjan},
      year={2026},
      eprint={2602.21766},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2602.21766}, 
}
```

---

## 👥 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Contribution areas:**
- New anomaly detection algorithms
- Additional robustness tests
- Performance optimizations
- Documentation improvements
- Bug fixes

---

## 👤 Contact

**Maintainer:** Mohamed Abdelmaksoud  
**Email:** mohamed@tu-berlin.de  
**Research Group:** D2IP @ TU Berlin

For questions, bug reports, or feature requests, please [open a GitHub issue](https://github.com/Maxoud99/RAMSeS/issues).

---

## 📝 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

**Important:** Individual datasets may have their own licenses. Please respect the original dataset licenses when using RAMSeS.

---

## 🙏 Acknowledgments

- The **Mononito** benchmark dataset team
- Contributors to the open-source anomaly detection libraries (PyOD, etc.)
- The D2IP research group at TU Berlin

---

**Happy Anomaly Detection! 🎯**
