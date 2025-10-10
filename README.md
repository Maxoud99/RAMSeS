# RAMSeS — Robust & Adaptive Model Selection for Time-Series Anomaly Detection Algorithms

**RAMSeS** is a research framework for **unsupervised time-series anomaly detection (TSAD)** that produces two deployable outputs:
1) An optimized **stacking ensemble** discovered by a Genetic Algorithm (GA), and  
2) A **top-ranked single model** selected via robust tests and **Linear Thompson Sampling** (LinTS).

This repository contains the end-to-end pipeline—data loading, model training/loading, robustness testing, online selection, and result aggregation—targeted at standard TSAD benchmarks.

> **Research group:** D2IP @ TU Berlin  
> **Status:** Research code (actively evolving)

--------------------------------------------------------

## ✨ Key Ideas

- **Two Branches:**  
  - **Ensemble branch:** GA searches subsets of base detectors; a fixed meta-learner (e.g., RF/LR/GBM/SVM) stacks their scores.  
  - **Single-model branch:** Combines (i) **Linear Thompson Sampling** with ε-greedy exploration, (ii) **GAN-based robustness** (borderline synthetic anomalies), (iii) **off-by-threshold** sensitivity tests, and (iv) **Monte-Carlo** noise stress-tests. A **Markov-chain rank aggregator** fuses rankings.

- **Online-ready:** Works with sliding windows and can update choices iteratively.

- **Reproducible outputs:** Human-readable summaries and plots are saved under `myresults/…`.

--------------------------------------------------------

## 🧱 Repository Layout (relevant parts)

```text
RAMSeS_framework/
├── app.py                              # main entrypoint (RAMSeS pipeline)
├── Datasets/
│   └── load.py                         # data loader wrapper
├── Metrics/
│   └── Ensemble_GA.py                  # GA, fitness, evaluation helpers
├── Model_Selection/
│   ├── Thompson_Sampling.py            # LinTS + sliding window utils
│   ├── rank_aggregation.py             # Markov-chain rank aggregator
│   ├── inject_anomalies.py             # anomaly injection utilities
│   └── Sensitivity_robustness/
│       ├── GAN_test.py                 # GAN-based robustness tests
│       ├── Monte_Carlo_Simulation.py   # noise stress-test
│       └── off_by_threshold_testing.py # borderline/off-by-threshold tests
├── Model_Training/
│   └── train.py                        # train base detectors, save .pth
└── Utils/
    └── utils.py                        # CLI arg parsing, misc
```



--------------------------------------------------------

## 📦 Datasets

We use the **Mononito** time-series repository introduced in the paper (arXiv:2210.01078).  
You can download the full dataset folder from Google Drive:
https://drive.google.com/drive/folders/1BLcaGm4bNSBueh3Hy_-dP1MKNhzfulwC?usp=share_link


Place it somewhere on disk (e.g., `~/Mononito/`) and point the CLI flag `--dataset_path` to it.  
If you use **SMD**, keep the structure consistent with `Datasets/load.py`.

> **Note on licensing:** Please follow the original dataset licenses and citations.

--------------------------------------------------------

## ⚙️ Requirements & Setup

We recommend Python **3.9+** and a recent **PyTorch**. GPU is optional.

```bash
# 1) Create and activate a virtual environment (conda shown; venv also fine)
conda create -n ramses python=3.10 -y
conda activate ramses

# 2) Install dependencies
pip install -r requirements.txt
# (If you don't have a requirements file yet, install torch + numpy + matplotlib + loguru + others your modules need.)

# 3) (Optional) Verify CUDA availability via torch.cuda.is_available()
python -c "import torch; print('CUDA:', torch.cuda.is_available())"


--------------------------------------------------------
🚀 Quickstart
1) Train base models (or reuse existing)
Model_Training/train.py saves detector checkpoints (.pth) under your chosen path.
In time_series_framework/app.py, training is triggered by:
model_trainer.train_models(model_architectures=args['model_architectures'])
Trained models are loaded by load_trained_models(..) from save_dir (hard-coded for now, see below).
If you already have trained models, you can skip training; just ensure save_dir and the .pth files align (see “Trained model path” below).

2) Run RAMSeS
The current pipeline order is:
Ensemble-GA → Thompson → GANs → Off-by-threshold → Monte Carlo → Rank Aggregations
Run:
cd RAMSes

python app.py \
  --dataset_path /ABS/PATH/TO/Mononito \
  --trained_model_path /ABS/PATH/TO/Mononito/trained_models \
  --downsampling 10 \
  --min_length 256 \
  --training_size 1.0 \
  --overwrite false \
  --verbose true \
  --model_architectures "CBLOF,LOF,LSTMVAE,MD,RM,DGHL, ..."

--------------------------------------------------------
Arguments (as parsed in Utils/utils.py:get_args_from_cmdline):
--dataset_path (required): root folder where datasets reside (e.g., Mononito).
--trained_model_path (required): base directory where training code writes/reads .pth.
--downsampling (int): downsample factor for series.
--min_length (int): minimum sequence length.
--training_size (float): portion of training split to use.
--overwrite (bool): whether to overwrite existing artifacts.
--verbose (bool): verbose logging.
--model_architectures (str): comma-separated list of base detectors to train.
Trained model path:
app.py currently loads model instances from a hard-coded save_dir at the top of the file:
save_dir = "/home/maxoud/projects/RAMS-TSAD/Mononito/trained_models/smd/machine-3-10/"
Either (a) change this to your path, or (b) symlink that path to your actual trained models folder. Make sure it contains *.pth files named like the entries in algorithm_list_instances (e.g., CBLOF_1.pth, LOF_3.pth, …).

--------------------------------------------------------
🧪 What gets produced
Rank summaries & aggregations:
myresults/robust_aggregated/{dataset}/{entity}/
  robust_aggregated_results_{dataset}_{entity}_{iteration}.txt
  new_robust_aggregated_results_{dataset}_{entity}_{iteration}.txt
Contains:
GAN, Borderline, Monte Carlo rankings
Robust rank aggregate (Markov-chain aggregation)
Final aggregation vs. Thompson sampling
Misclassification summaries (per-window), if the real-time loop runs
Diagnostics & plots:
myresults/GA_Ens/{dataset}/{entity}/
  ensemble_scores_{dataset}_{entity}_Data_vs_anomalies_[...].png
Shows injected anomaly segments and scores for quick visual checks.

--------------------------------------------------------
🧩 Base Detectors & Instances
Defined in app.py:
algorithm_list = ['DGHL', 'LSTMVAE', 'MD', 'RM', 'LOF', 'CBLOFd', ...]
algorithm_list_instances = [
  'CBLOF_1', 'CBLOF_2', 'CBLOF_3', 'CBLOF_4',
  'DGHL_1', 'DGHL_2', 'DGHL_3', 'DGHL_4',
  'LOF_1', 'LOF_2', 'LOF_3', 'LOF_4',
  'LSTMVAE_1', 'LSTMVAE_2', 'LSTMVAE_3', 'LSTMVAE_4',
  'MD_1',
  'RM_1', 'RM_2', 'RM_3', ...
]
Your training should produce checkpoints named exactly like the entries in algorithm_list_instances. The GA operates on these instances; the meta-learner type is configured inside app.py (currently 'rf' in run_model_selection_algorithms_1).

--------------------------------------------------------
🔁 Sliding Windows & Online Mode
We use:
initialize_sliding_windows(data, targets, mask, window_size, stride)
In app.py, iterations = 1 by default (single pass). Increase it for real-time evaluation; the loop will:
evaluate current best single model on the new window,
evaluate the GA-selected ensemble fitness,
write misclassification summaries, and
update the selections via a fresh pass through the pipeline.

--------------------------------------------------------
🧪 Reproducibility Tips
Fix random seeds (GA, Monte-Carlo, GAN sampling, Thompson).
Log/save all hyper-params along with results (consider dumping the parsed CLI args).
Keep dataset_path, trained_model_path, and save_dir consistent and absolute.

--------------------------------------------------------
❓ Troubleshooting
“Model X not found in save_dir”
Ensure save_dir points to the directory containing .pth named like algorithm_list_instances.
If you trained to a different layout, adjust save_dir or rename/copy files.
Empty entities in train/test
Verify --dataset_path points to the Mononito root and matches the layout expected by Datasets/load.py.
Matplotlib/Display issues on servers
Set a non-interactive backend: export MPLBACKEND=Agg or add before plotting:
import matplotlib
matplotlib.use("Agg")

--------------------------------------------------------
📚 Citation
If you use RAMSeS / RAMS-TSAD in your work, please cite our paper:
# @inproceedings{YOUR_KEY_HERE,
#   title        = {RAMSeS: Robust and Adaptive Model Selection for Time-Series Anomaly Detection},
#   author       = {Your Name and Coauthors},
#   booktitle    = {ICDE 2026 (to appear)},
#   year         = {2026},
#   note         = {Code available at: https://github.com/...}
# }
If you use the Mononito datasets, please also cite the original authors (see their paper: arXiv:2210.01078).
--------------------------------------------------------
👤 Contact
Maintainer: Mohamed Abdelmaksoud (mohamed@tu-berlin.de)
For questions/bug reports: please open a GitHub issue.

--------------------------------------------------------
📝 License
This repository is released for research purposes. Check dataset licenses for any additional restrictions. See LICENSE for details (or choose a suitable OSI license and update this section).
