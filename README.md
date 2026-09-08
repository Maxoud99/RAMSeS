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
- Sliding window processing with configurable update intervals
- Multiple strategies: adaptive re-optimization, fixed-best, or fixed-random
- Optional regime shift injection for testing distribution changes
- Real-time model selection updates

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
- **TensorFlow** 2.18 (CPU version)
- **scikit-learn** 1.7 (ML algorithms)
- **PyOD** 3.6 (outlier detection)
- **NumPy**, **Pandas**, **Matplotlib** (data processing & visualization)
- **loguru** (structured logging)

- **Flask** 3.1 (serves the local WebUI)

All 26 dependencies are pinned in `requirements.txt`; `environment.yml`
installs from it, so the conda and pip routes cannot drift apart.

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
  --dataset <name>                # Dataset name (e.g. SKAB, SMD, UCR)
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
  --dataset SKAB \
  --entity 5 \
  --enable_online \
  --update_interval 5 \
  --iteration 5 \
  --strategy adaptive \
  --inject_online_regime \
  --max_online_windows 100
```

**Online mode parameters:**
- `--enable_online` — Enable online phase processing
- `--update_interval <n>` — Re-optimize every n windows (default: 5)
- `--iteration <n>` — Number of iterations for window sizing (default: 5)
- `--strategy <mode>` — Selection strategy:
  - `adaptive` — Re-optimize model selection periodically
  - `fixed-best` — Use best offline model throughout
  - `fixed-random` — Use random model (baseline)
- `--inject_online_regime` — Inject regime shifts in online data for testing
- `--max_online_windows <n>` — Limit number of online windows processed

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

## 🧪 Reproducibility Tips

- **Fix random seeds** for all stochastic components (GA, Monte Carlo, GAN, Thompson Sampling)
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
