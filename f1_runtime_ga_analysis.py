# runtime_ga_analysis.py
# HARD-CODED, RUN-BUTTON READY (no configs, no CLI).
# Produces under output/ga_runtime_YYYYMMDD-HHMMSS/:
#   - ga_runtime_results.csv
#   - figure_runtime_f1_icde.png  (2x3: Runtime row + F1 row)
#   - summary.txt

import os
import shutil
import time
import math
import copy
import random
import logging
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Callable

import numpy as np
import torch as t
import matplotlib.pyplot as plt

# -----------------------------
# HARD-CODED CONFIGURATIONS
# -----------------------------

# EXACT PATHS (as requested)
TRAINED_MODELS_DIR = "/home/maxoud/projects/RAMS-TSAD/Mononito/trained_models/skab/1"
DATASET_FILE       = "/home/maxoud/projects/RAMS-TSAD/Mononito/datasets/skab/1.csv"

# Dataset identity (we’ll stage the single CSV into a folder layout the loader expects)
DATASET_NAME = "skab"
ENTITY       = "1"

# Preprocessing
DOWNSAMPLING = 10
MIN_LENGTH   = 256
NORMALIZE    = True
VERBOSE      = False

# GA sweeps (factorized, ICDE-friendly)
POP_SET = [2, 5, 10, 20, 50, 100]
GEN_SET = [1, 2, 5, 10, 20, 50, 100, 500]
MU_SET  = [0.0, 0.05, 0.2, 0.5, 1.0]
META_SET = ['rf', 'lr', 'svm']

# Workloads for meta comparison (label -> (P, G))
WORKLOADS = {
    "P= 5, G= 20":   (5, 20),
    "P= 20, G= 100": (20, 100),
    "P= 50, G= 500":  (50, 500),
}

REPEATS = 3                      # median + IQR
ANOMALIES = ['spikes']           # single fixed injection
SEED_GLOBAL = 1337               # fixed seed for anomaly injection

# Base model instances (deduplicated but order-preserving) — use only what you actually have
ALGORITHM_LIST_INSTANCES = list(dict.fromkeys([
    'LOF_1', 'LOF_2', 'LOF_3', 'LOF_4',
    'NN_1', 'NN_2', 'NN_3',
    'RNN_1', 'RNN_2', 'RNN_3', 'RNN_4',
    # add more if available in TRAINED_MODELS_DIR
]))

# -----------------------------
# PROJECT IMPORTS (no configs)
# -----------------------------
from Datasets.load import load_data
from Metrics.Ensemble_GA import genetic_algorithm
from Model_Selection.inject_anomalies import Inject

# -------------------
# Logging & utilities
# -------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("GA-Runtime")

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        t.manual_seed(seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(seed)
    except Exception:
        pass

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def timestamp_dir(base="output", prefix="ga_runtime") -> str:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(base, f"{prefix}_{ts}")
    ensure_dir(out)
    return out

def stage_single_csv_for_loader(csv_file: str, out_root: str, dataset_name: str, entity: str) -> str:
    """
    Your loader expects: <root>/<dataset>/<entity>/[*.csv]
    But you have a single CSV file. We create that structure under out_root and copy the CSV there.
    Returns the staging root_dir to pass into load_data().
    """
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"DATASET_FILE not found: {csv_file}")
    ds_dir = os.path.join(out_root, dataset_name, entity)
    ensure_dir(ds_dir)
    # Copy as-is (name does not matter; loader lists all .csv in the folder)
    dst = os.path.join(ds_dir, os.path.basename(csv_file))
    if os.path.abspath(csv_file) != os.path.abspath(dst):
        shutil.copy2(csv_file, dst)
    log.info("Staged CSV -> %s", ds_dir)
    return out_root  # this becomes root_dir for load_data

def load_trained_models(model_names: List[str], models_dir: str) -> Dict[str, Any]:
    """
    Tolerant loader: loads any .pth that exists; warns about missing.
    """
    trained = {}
    missing = []
    for name in model_names:
        path = os.path.join(models_dir, f"{name}.pth")
        if not os.path.exists(path):
            missing.append(name)
            continue
        with open(path, 'rb') as fh:
            model = t.load(fh, weights_only=False)
            try:
                model.eval()
            except Exception:
                pass
            trained[name] = model
    if missing:
        log.warning("Missing %d checkpoints (showing up to 5): %s", len(missing), missing[:5])
    log.info("Loaded %d/%d trained models from %s", len(trained), len(model_names), models_dir)
    if len(trained) == 0:
        raise FileNotFoundError(
            f"No .pth files loaded from {models_dir}. "
            f"Expected some of: {[n+'.pth' for n in model_names[:6]]} ..."
        )
    return trained

@dataclass
class RunResult:
    dataset: str
    entity: str
    P: int
    G: int
    mu: float
    meta: str
    run_idx: int
    runtime_s: float
    best_f1: float
    best_pr_auc: float
    ensemble_size: int
    error: str = ""

# ----------------------------
# GA timing for one config/run
# ----------------------------

def time_ga_once(
    dataset: str,
    entity: str,
    train_data,
    test_data,
    model_names: List[str],
    trained_models: Dict[str, Any],
    population_size: int,
    generations: int,
    meta_model_type: str,
    mutation_rate: float,
    seed: int
) -> Tuple[float, float, float, int]:
    """
    Returns: (runtime_s, best_f1, best_pr_auc, ensemble_size)
    """
    train_copy = copy.deepcopy(train_data)
    test_copy  = copy.deepcopy(test_data)

    set_all_seeds(seed)

    start = time.perf_counter()
    best_ensemble, best_f1, best_pr_auc, best_fitness, \
    individual_predictions, base_model_predictions_train, base_model_predictions_test, \
    y_true_train, y_true_test, chosen_meta = genetic_algorithm(
        dataset, entity,
        train_copy, test_copy,
        model_names, trained_models,
        population_size=population_size,
        generations=generations,
        meta_model_type=meta_model_type,
        mutation_rate=mutation_rate,
    )
    try:
        if t.cuda.is_available():
            t.cuda.synchronize()
    except Exception:
        pass
    end = time.perf_counter()

    ensemble_size = len(best_ensemble) if isinstance(best_ensemble, (list, tuple, set)) else int(best_ensemble is not None)
    return (end - start), float(best_f1), float(best_pr_auc), ensemble_size

# ---------------
# Plotting helpers
# ---------------

def icde_style():
    plt.rcParams.update({
        "figure.figsize": (13, 8.6),   # 2 rows x 3 cols; wide but taller
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "errorbar.capsize": 3,
    })

def compute_group_stats(
    rows: List[RunResult],
    key_fields: Tuple[str, ...],
    value_selector: Callable[[RunResult], float]
):
    """
    Group by key_fields; compute median and IQR for the metric given by value_selector.
    """
    groups: Dict[Tuple, List[RunResult]] = {}
    for r in rows:
        k = tuple(getattr(r, f) for f in key_fields) if key_fields else ("_all",)
        groups.setdefault(k, []).append(r)

    out = []
    for k, vs in groups.items():
        vals = np.array([value_selector(v) for v in vs if math.isfinite(value_selector(v))])
        if len(vals) == 0:
            med = q25 = q75 = float('nan')
        else:
            med = float(np.median(vals))
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
        rec = {}
        if key_fields:
            rec.update({fld: val for fld, val in zip(key_fields, k)})
        rec.update(dict(n=len(vs), median=med, q25=q25, q75=q75))
        out.append(rec)
    return out

def save_csv(rows: List[RunResult], path: str) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset","entity","P","G","mu","meta","run_idx",
            "runtime_s","best_f1","best_pr_auc","ensemble_size","error"
        ])
        for r in rows:
            w.writerow([
                r.dataset, r.entity, r.P, r.G, r.mu, r.meta, r.run_idx,
                f"{r.runtime_s:.6f}", f"{r.best_f1:.6f}", f"{r.best_pr_auc:.6f}", r.ensemble_size, r.error
            ])

def _scatter_with_fit(ax, x, y, y_lo, y_hi, label=None, show_r2=True, logx=False):
    ax.errorbar(x, y, yerr=[y_lo, y_hi], fmt='o', label=label)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() >= 2:
        coef = np.polyfit(x[good], y[good], deg=1)
        xs = np.linspace(x[good].min(), x[good].max(), 200)
        ax.plot(xs, np.polyval(coef, xs))
        if show_r2:
            yhat = np.polyval(coef, x[good])
            ss_res = np.sum((y[good] - yhat)**2)
            ss_tot = np.sum((y[good] - np.mean(y[good]))**2)
            r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)
            ax.text(0.05, 0.95, f"R²={r2:.3f}", transform=ax.transAxes, ha="left", va="top")
    if logx:
        ax.set_xscale("log")

def plot_runtime_and_f1(out_dir: str, rows: List[RunResult]) -> None:
    icde_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    # Row 1: Runtime; Row 2: F1
    axA, axB, axC = axes[0]
    axD, axE, axF = axes[1]

    # ---------------------------
    # (A) Runtime vs Work, μ=0.2, meta=rf
    # ---------------------------
    subset = [r for r in rows if r.mu == 0.2 and r.meta == 'rf']
    stats = compute_group_stats(subset, key_fields=("P","G"), value_selector=lambda r: r.runtime_s)
    stats = sorted(stats, key=lambda d: (d["P"]*d["G"], d["P"], d["G"]))
    x_work = np.array([d["P"]*d["G"] for d in stats], dtype=float)
    y_med  = np.array([d["median"] for d in stats], dtype=float)
    y_lo   = np.array([d["median"] - d["q25"] for d in stats], dtype=float)
    y_hi   = np.array([d["q75"] - d["median"] for d in stats], dtype=float)
    _scatter_with_fit(axA, x_work, y_med, y_lo, y_hi, logx=True)
    axA.set_xlabel("Work (population × generations)")
    axA.set_ylabel("Runtime (s)")
    axA.set_title("A) Runtime vs Work (μ=0.2, meta=rf)")

    # ---------------------------
    # (B) Runtime vs μ, lines per meta at fixed workload (P=50,G=10)
    # ---------------------------
    P_fix, G_fix = 50, 10
    for meta in META_SET:
        sub = [r for r in rows if r.P == P_fix and r.G == G_fix and r.meta == meta]
        statB = compute_group_stats(sub, key_fields=("mu",), value_selector=lambda r: r.runtime_s)
        statB = sorted(statB, key=lambda d: d["mu"])
        x_mu = np.array([d["mu"] for d in statB], float)
        y_med = np.array([d["median"] for d in statB], float)
        y_lo  = np.array([d["median"] - d["q25"] for d in statB], float)
        y_hi  = np.array([d["q75"] - d["median"] for d in statB], float)
        if len(x_mu) > 0:
            axB.errorbar(x_mu, y_med, yerr=[y_lo, y_hi], marker='o', label=meta)
    axB.set_xlabel("Mutation rate (μ)")
    axB.set_ylabel("Runtime (s)")
    axB.set_title("B) Runtime vs Mutation (P=50, G=10)")
    # axB.legend(title="meta")
    axB.legend(
    title="meta",
    ncols=3,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),   # centered, just below the plot
    frameon=True,
    borderaxespad=0.0
    )
    # ---------------------------
    # (C) Runtime vs meta at 3 workloads (μ=0.2)
    # ---------------------------
    x_positions = np.arange(len(WORKLOADS))
    width = 0.25
    offset_map = {'rf': -width, 'lr': 0.0, 'svm': width}
    for meta in META_SET:
        xs, ys, yerr_lo, yerr_hi = [], [], [], []
        for i, (label, (P, G)) in enumerate(WORKLOADS.items()):
            sub = [r for r in rows if r.P == P and r.G == G and r.mu == 0.2 and r.meta == meta]
            statC = compute_group_stats(sub, key_fields=tuple(), value_selector=lambda r: r.runtime_s)
            if len(statC) == 1:
                s = statC[0]
                xs.append(x_positions[i] + offset_map[meta])
                ys.append(s["median"])
                yerr_lo.append(s["median"] - s["q25"])
                yerr_hi.append(s["q75"] - s["median"])
            else:
                xs.append(x_positions[i] + offset_map[meta]); ys.append(np.nan); yerr_lo.append(0.0); yerr_hi.append(0.0)
        axC.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt='o', label=meta)
    axC.set_xticks(x_positions, list(WORKLOADS.keys()))
    axC.set_xlabel("Workload")
    axC.set_ylabel("Runtime (s)")
    axC.set_title("C) Runtime vs Meta at 3 Workloads (μ=0.2)")
    # axC.legend(title="meta", ncols=3, loc="upper left")
    axC.legend(
    title="meta",
    ncols=3,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),   # centered, just below the plot
    frameon=True,
    borderaxespad=0.0)
    # ---------------------------
    # (D) F1 vs Work, μ=0.2, meta=rf  (same style)
    # ---------------------------
    stats = compute_group_stats(subset, key_fields=("P","G"), value_selector=lambda r: r.best_f1)
    stats = sorted(stats, key=lambda d: (d["P"]*d["G"], d["P"], d["G"]))
    x_work = np.array([d["P"]*d["G"] for d in stats], dtype=float)
    y_med  = np.array([d["median"] for d in stats], dtype=float)
    y_lo   = np.array([d["median"] - d["q25"] for d in stats], dtype=float)
    y_hi   = np.array([d["q75"] - d["median"] for d in stats], dtype=float)
    _scatter_with_fit(axD, x_work, y_med, y_lo, y_hi, logx=True)
    axD.set_xlabel("Work (population × generations)")
    axD.set_ylabel("F1")
    axD.set_title("D) F1 vs Work (μ=0.2, meta=rf)")

    # ---------------------------
    # (E) F1 vs μ, lines per meta at fixed workload (P=50,G=10)
    # ---------------------------
    for meta in META_SET:
        sub = [r for r in rows if r.P == P_fix and r.G == G_fix and r.meta == meta]
        statE = compute_group_stats(sub, key_fields=("mu",), value_selector=lambda r: r.best_f1)
        statE = sorted(statE, key=lambda d: d["mu"])
        x_mu = np.array([d["mu"] for d in statE], float)
        y_med = np.array([d["median"] for d in statE], float)
        y_lo  = np.array([d["median"] - d["q25"] for d in statE], float)
        y_hi  = np.array([d["q75"] - d["median"] for d in statE], float)
        if len(x_mu) > 0:
            axE.errorbar(x_mu, y_med, yerr=[y_lo, y_hi], marker='o', label=meta)
    axE.set_xlabel("Mutation rate (μ)")
    axE.set_ylabel("F1")
    axE.set_title("E) F1 vs Mutation (P=50, G=10)")
    # axE.legend(title="meta")
    axE.legend(
    title="meta",
    ncols=3,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),   # centered, just below the plot
    frameon=True,
    borderaxespad=0.0)

    # ---------------------------
    # (F) F1 vs meta at 3 workloads (μ=0.2)
    # ---------------------------
    for meta in META_SET:
        xs, ys, yerr_lo, yerr_hi = [], [], [], []
        for i, (label, (P, G)) in enumerate(WORKLOADS.items()):
            sub = [r for r in rows if r.P == P and r.G == G and r.mu == 0.2 and r.meta == meta]
            statF = compute_group_stats(sub, key_fields=tuple(), value_selector=lambda r: r.best_f1)
            if len(statF) == 1:
                s = statF[0]
                xs.append(x_positions[i] + offset_map[meta])
                ys.append(s["median"])
                yerr_lo.append(s["median"] - s["q25"])
                yerr_hi.append(s["q75"] - s["median"])
            else:
                xs.append(x_positions[i] + offset_map[meta]); ys.append(np.nan); yerr_lo.append(0.0); yerr_hi.append(0.0)
        axF.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt='o', label=meta)
    axF.set_xticks(x_positions, list(WORKLOADS.keys()))
    axF.set_xlabel("Workload")
    axF.set_ylabel("F1")
    axF.set_title("F) F1 vs Meta at 3 Workloads (μ=0.2)")
  # Put legend below the axis to avoid covering data
    axF.legend(
        title="meta",
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),   # centered, just below the plot
        frameon=True,
        borderaxespad=0.0
    )


    plt.tight_layout(rect=[0, 0.08, 1, 1], h_pad=1.0)
    figpath = os.path.join(out_dir, "figure_runtime_f1_icde.png")
    plt.savefig(figpath, dpi=300)
    log.info("Saved figure: %s", figpath)

# -----------------
# Experiment runner
# -----------------

def main():
    # Output dir
    out_dir = timestamp_dir(base="output", prefix="ga_runtime")
    log.info("Output directory: %s", out_dir)

    # Stage the single CSV into a directory structure the loader expects
    staging_root = os.path.join(out_dir, "_dataset_staging")
    staged_root_dir = stage_single_csv_for_loader(
        csv_file=DATASET_FILE,
        out_root=staging_root,
        dataset_name=DATASET_NAME,
        entity=ENTITY,
    )

    # Load data (train & test) using staged root_dir
    log.info("Loading dataset=%s entity=%s (downsampling=%d, min_length=%d)", DATASET_NAME, ENTITY, DOWNSAMPLING, MIN_LENGTH)
    train_data = load_data(
        dataset=DATASET_NAME, group='train',
        entities=ENTITY, downsampling=DOWNSAMPLING,
        min_length=MIN_LENGTH, root_dir=staged_root_dir, normalize=NORMALIZE, verbose=VERBOSE
    )
    test_data = load_data(
        dataset=DATASET_NAME, group='test',
        entities=ENTITY, downsampling=DOWNSAMPLING,
        min_length=MIN_LENGTH, root_dir=staged_root_dir, normalize=NORMALIZE, verbose=VERBOSE
    )
    if not getattr(train_data, "entities", None) or not getattr(test_data, "entities", None):
        raise RuntimeError("Failed to load train/test. Check DATASET_NAME, staged_root_dir, and ENTITY.")

    # Fixed anomaly injection reused for all runs
    set_all_seeds(SEED_GLOBAL)
    train_inj, _ = Inject(copy.deepcopy(train_data), ANOMALIES)
    test_inj, _  = Inject(copy.deepcopy(test_data), ANOMALIES)
    log.info("Injected anomalies: %s (fixed for all runs)", ANOMALIES)

    # Load trained models (tolerant)
    log.info("Loading trained models from: %s", TRAINED_MODELS_DIR)
    trained_models = load_trained_models(ALGORITHM_LIST_INSTANCES, TRAINED_MODELS_DIR)

    # Run experiments
    results: List[RunResult] = []

    # E1: Runtime & F1 vs Work (μ=0.2, meta='rf')
    log.info("E1: Runtime/F1 vs Work (μ=0.2, meta=rf) over %d×%d grid with %d repeats", len(POP_SET), len(GEN_SET), REPEATS)
    for P in POP_SET:
        for G in GEN_SET:
            for run_idx in range(REPEATS):
                try:
                    rt, f1, pr, ensz = time_ga_once(
                        DATASET_NAME, ENTITY, train_inj, test_inj,
                        ALGORITHM_LIST_INSTANCES, trained_models,
                        population_size=P, generations=G,
                        meta_model_type='rf', mutation_rate=0.2,
                        seed=1000 + run_idx
                    )
                    results.append(RunResult(DATASET_NAME, ENTITY, P, G, 0.2, 'rf', run_idx, rt, f1, pr, ensz))
                except Exception as e:
                    results.append(RunResult(DATASET_NAME, ENTITY, P, G, 0.2, 'rf', run_idx, float('nan'), float('nan'), float('nan'), 0, error=str(e)))
                    log.exception("E1 failure at P=%d, G=%d, run=%d", P, G, run_idx)

    # E2: Runtime & F1 vs Mutation at fixed workload (P=50, G=10) with meta ∈ {rf, lr, svm}
    P_fix, G_fix = 50, 10
    log.info("E2: Runtime/F1 vs μ at (P=%d, G=%d) for meta in %s with %d repeats", P_fix, G_fix, META_SET, REPEATS)
    for meta in META_SET:
        for mu in MU_SET:
            for run_idx in range(REPEATS):
                try:
                    rt, f1, pr, ensz = time_ga_once(
                        DATASET_NAME, ENTITY, train_inj, test_inj,
                        ALGORITHM_LIST_INSTANCES, trained_models,
                        population_size=P_fix, generations=G_fix,
                        meta_model_type=meta, mutation_rate=mu,
                        seed=2000 + run_idx
                    )
                    results.append(RunResult(DATASET_NAME, ENTITY, P_fix, G_fix, mu, meta, run_idx, rt, f1, pr, ensz))
                except Exception as e:
                    results.append(RunResult(DATASET_NAME, ENTITY, P_fix, G_fix, mu, meta, run_idx, float('nan'), float('nan'), float('nan'), 0, error=str(e)))
                    log.exception("E2 failure at meta=%s, mu=%.3f, run=%d", meta, mu, run_idx)

    # E3: Runtime & F1 vs Meta at three workloads (μ=0.2)
    log.info("E3: Runtime/F1 vs meta at 3 workloads (μ=0.2) with %d repeats", REPEATS)
    for label, (P, G) in WORKLOADS.items():
        for meta in META_SET:
            for run_idx in range(REPEATS):
                try:
                    rt, f1, pr, ensz = time_ga_once(
                        DATASET_NAME, ENTITY, train_inj, test_inj,
                        ALGORITHM_LIST_INSTANCES, trained_models,
                        population_size=P, generations=G,
                        meta_model_type=meta, mutation_rate=0.2,
                        seed=3000 + run_idx
                    )
                    results.append(RunResult(DATASET_NAME, ENTITY, P, G, 0.2, meta, run_idx, rt, f1, pr, ensz))
                except Exception as e:
                    results.append(RunResult(DATASET_NAME, ENTITY, P, G, 0.2, meta, run_idx, float('nan'), float('nan'), float('nan'), 0, error=str(e)))
                    log.exception("E3 failure at workload=%s, meta=%s, run=%d", label, meta, run_idx)

    # Persist CSV + Figure + Summary
    csv_path = os.path.join(out_dir, "ga_runtime_results.csv")
    save_csv(results, csv_path)
    log.info("Saved CSV: %s (%d rows)", csv_path, len(results))

    plot_runtime_and_f1(out_dir, results)

    txt_path = os.path.join(out_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write("GA Runtime & F1 Study (RAMSeS) — Summary\n")
        f.write(f"Timestamp: {dt.datetime.now().isoformat()}\n")
        f.write(f"Dataset: {DATASET_NAME} / entity={ENTITY}\n")
        f.write(f"Data file: {DATASET_FILE}\n")
        f.write(f"Anomalies injected (fixed): {ANOMALIES}\n")
        f.write(f"Repeats per config: {REPEATS}\n")
        f.write(f"Models dir: {TRAINED_MODELS_DIR}\n\n")

        # E1 medians across (P,G) for Runtime and F1
        subset = [r for r in results if r.mu == 0.2 and r.meta == 'rf']
        stats_rt = compute_group_stats(subset, key_fields=("P","G"), value_selector=lambda r: r.runtime_s)
        stats_rt = [s for s in stats_rt if math.isfinite(s["median"])]
        stats_f1 = compute_group_stats(subset, key_fields=("P","G"), value_selector=lambda r: r.best_f1)
        stats_f1 = [s for s in stats_f1 if math.isfinite(s["median"])]

        if stats_rt:
            best_rt = sorted(stats_rt, key=lambda d: d["median"])[:5]
            worst_rt = sorted(stats_rt, key=lambda d: d["median"])[-5:]
            f.write("E1 — Fastest 5 configs by median runtime:\n")
            for b in best_rt:
                f.write(f"  P={b['P']:>3}, G={b['G']:>4}, work={b['P']*b['G']:>5}, "
                        f"runtime_med={b['median']:.4f}s, IQR=[{b['q25']:.4f},{b['q75']:.4f}]\n")
            f.write("E1 — Slowest 5 configs by median runtime:\n")
            for w in worst_rt:
                f.write(f"  P={w['P']:>3}, G={w['G']:>4}, work={w['P']*w['G']:>5}, "
                        f"runtime_med={w['median']:.4f}s, IQR=[{w['q25']:.4f},{w['q75']:.4f}]\n")
            f.write("\n")

        if stats_f1:
            best_f1 = sorted(stats_f1, key=lambda d: d["median"], reverse=True)[:5]
            f.write("E1 — Top 5 configs by median F1:\n")
            for b in best_f1:
                f.write(f"  P={b['P']:>3}, G={b['G']:>4}, work={b['P']*b['G']:>5}, "
                        f"F1_med={b['median']:.4f}, IQR=[{b['q25']:.4f},{b['q75']:.4f}]\n")
            f.write("\n")

        # Failures
        failures = [r for r in results if not math.isfinite(r.runtime_s)]
        f.write(f"Failures (runtime NaN): {len(failures)}\n")
        if failures:
            f.write("Example errors (up to 5):\n")
            for r in failures[:5]:
                f.write(f"  P={r.P}, G={r.G}, mu={r.mu}, meta={r.meta}, err={r.error[:200]}\n")

    log.info("Saved summary: %s", txt_path)
    log.info("Done. All artifacts in: %s", out_dir)

if __name__ == "__main__":
    main()
