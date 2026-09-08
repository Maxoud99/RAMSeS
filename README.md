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
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment specification
├── LICENSE                             # Apache 2.0 license
│
├── Algorithms/                         # Base anomaly detector implementations
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
│   ├── nearest_neighbors.py            # k-Nearest Neighbors
│   ├── rnn.py                          # Recurrent Neural Network
│   ├── running_mean.py                 # Running Mean
│   └── sos.py                          # Stochastic Outlier Selection
│
├── Configs/                            # Configuration files
│   ├── config.yml                      # Default configuration
│   └── custom_config.yml               # Custom configuration template
│
├── Datasets/                           # Data loading utilities
│   ├── dataset.py                      # Dataset class definitions
│   └── load.py                         # Data loader implementations
│
├── Metrics/                            # Evaluation and GA components
│   ├── Ensemble_GA.py                  # Genetic Algorithm for ensemble optimization
│   ├── metrics.py                      # Performance metrics (F1, PR-AUC, etc.)
│   └── ranking_metrics.py              # Ranking evaluation utilities
│
├── Model_Selection/                    # Model selection algorithms
│   ├── Thompson_Sampling.py            # Linear Thompson Sampling + sliding windows
│   ├── rank_aggregation.py             # Markov-chain rank aggregation
│   ├── inject_anomalies.py             # Synthetic anomaly injection
│   └── Sensitivity_robustness/
│       ├── GAN_test.py                 # GAN-based robustness testing
│       ├── Monte_Carlo_Simulation.py   # Monte Carlo noise stress tests
│       └── off_by_threshold_testing.py # Borderline sensitivity analysis
│
├── Model_Training/                     # Model training and management
│   ├── train.py                        # Training orchestration
│   ├── trainer.py                      # Training logic
│   └── hyperparameter_grids.py         # Hyperparameter configurations
│
├── Utils/                              # Utility functions
│   ├── utils.py                        # CLI argument parsing, misc utilities
│   ├── config.py                       # Configuration file parser
│   ├── plotting.py                     # Visualization utilities
│   └── results_formatter.py            # Results formatting and export
│
└── testbed/                            # Batch evaluation configurations
    └── file_list/                      # CSV files listing datasets for batch runs
        ├── test_single.csv             # Single dataset test
        ├── test_m_skab.csv             # SKAB multivariate datasets
        └── test_u_ucr_anomaly_archive.csv  # UCR univariate datasets
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
conda activate ..

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
- **PyTorch** 2.5+ (with optional CUDA support)
- **TensorFlow** 2.18+ (CPU version)
- **scikit-learn** 1.7+ (ML algorithms)
- **PyOD** 2.0+ (outlier detection)
- **NumPy**, **Pandas**, **Matplotlib** (data processing & visualization)
- **loguru** (structured logging)

See `requirements.txt` for the complete dependency list.

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
  --config Configs/config.yml \
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
  --config Configs/config.yml \
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

## 🎯 Configuration Options

### Command-Line Arguments

```bash
python app.py \
  --config <path>           # Path to config file (default: Configs/config.yml)
  --dataset <name>          # Dataset name (e.g., SKAB, SMD, UCR)
  --entity <id>             # Entity ID within dataset
  --parallel <true|false>   # Enable parallel model selection
```

### Online Mode (Adaptive Selection)

Enable online learning and adaptive model selection:

```bash
python app.py \
  --config Configs/config.yml \
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

RAMSeS includes the following base detectors:

**Statistical Methods:**
- `LOF` — Local Outlier Factor
- `CBLOF` — Cluster-Based Local Outlier Factor
- `COF` — Connectivity-Based Outlier Factor
- `KDE` — Kernel Density Estimation
- `NN` — k-Nearest Neighbors (Nearest Neighbors)
- `SOS` — Stochastic Outlier Selection

**Classical Methods:**
- `MD` — Mean Deviation
- `RM` — Running Mean
- `ABOD` — Angle-Based Outlier Detection

**Deep Learning Methods:**
- `LSTMVAE` — LSTM Variational Autoencoder
- `RNN` — Recurrent Neural Network
- `DGHL` — Deep Generative Hierarchical Learning
- `ALAD` — Adversarial Learned Anomaly Detection
- `AnomalyTransformer` — Transformer-based detector

**Configuration in `Configs/config.yml`:**
```yaml
model_architectures: 'all'  # Train all available algorithms
# OR specify subset:
model_architectures: 'LOF,CBLOF,NN,LSTMVAE'
```

Each algorithm can have multiple instances with different hyperparameters (e.g., `LOF_1`, `LOF_2`, `LOF_3`) for diversity in ensemble construction.

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
python app.py --config Configs/config.yml --dataset SKAB --entity 5
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
