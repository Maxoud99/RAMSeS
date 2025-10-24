# runtime_ga_analysis.py
# HARD-CODED, RUN-BUTTON READY (no configs, no CLI).
# Produces under output/ga_runtime_YYYYMMDD-HHMMSS/:
#   - ga_runtime_results.csv
#   - figure1_runtime_icde.png  (3 subplots: Work, Mutation, Meta)
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
from typing import Dict, List, Tuple, Any

import numpy as np
import torch as t
import matplotlib.pyplot as plt

# -----------------------------
# HARD-CODED CONFIGURATIONS
# -----------------------------

# EXACT PATHS (as requested)
TRAINED_MODELS_DIR = "/home/maxoud/projects/RAMSeS/Mononito/trained_models/anomaly_archive/011_UCR_Anomaly_DISTORTEDECG1_10000_11800_12100"
DATASET_FILE       = "/home/maxoud/projects/RAMSeS/Mononito/datasets/Anomaly_Archive/011_UCR_Anomaly_DISTORTEDECG1_10000_11800_12100.txt"

# Dataset identity (we’ll stage the single CSV into a folder layout the loader expects)
DATASET_NAME = "Anomaly_Archive"
ENTITY       = "011_UCR_Anomaly_DISTORTEDECG1_10000_11800_12100"

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
    "light (P= 5, G= 20)":  (5, 10),
    "medium (P= 50, G= 100)": (50, 20),
    "heavy (P= 50, G= 500)":  (100, 50),
}

REPEATS = 3                      # median + IQR
ANOMALIES = ['spikes']           # single fixed injection
SEED_GLOBAL = 1337               # fixed seed for anomaly injection

# Base model instances (deduplicated but order-preserving)
ALGORITHM_LIST_INSTANCES = list(dict.fromkeys([
    'LOF_1', 'LOF_2', 'LOF_3', 'LOF_4',
    'NN_1', 'NN_2', 'NN_3',
    'RNN_1', 'RNN_2', 'RNN_3', 'RNN_4'
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
    ds_dir = os.path.join(out_root, dataset_name)
    ensure_dir(ds_dir)
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
        "figure.figsize": (12, 4.8),   # wide but short, 3 subplots
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "errorbar.capsize": 3,
    })

def compute_group_stats(rows: List[RunResult], key_fields: Tuple[str, ...]):
    """
    Group by key_fields; compute median runtime, q25, q75.
    """
    groups = {}
    for r in rows:
        k = tuple(getattr(r, f) for f in key_fields) if key_fields else ("_all",)
        groups.setdefault(k, []).append(r)

    out = []
    for k, vs in groups.items():
        runtimes = np.array([v.runtime_s for v in vs if math.isfinite(v.runtime_s)])
        if len(runtimes) == 0:
            med = q25 = q75 = float('nan')
        else:
            med = float(np.median(runtimes))
            q25 = float(np.percentile(runtimes, 25))
            q75 = float(np.percentile(runtimes, 75))
        rec = {}
        if key_fields:
            rec.update({fld: val for fld, val in zip(key_fields, k)})
        rec.update(dict(
            n=len(vs),
            runtime_median=med,
            runtime_q25=q25,
            runtime_q75=q75
        ))
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

def plot_figure_1(output_dir: str, rows: List[RunResult]) -> None:
    icde_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axA, axB, axC = axes

    # (A) Runtime vs (P * G), μ=0.2, meta='rf'
    subset_A = [r for r in rows if r.mu == 0.2 and r.meta == 'rf']
    stats_A = compute_group_stats(subset_A, key_fields=("P","G"))
    stats_A = sorted(stats_A, key=lambda d: (d["P"]*d["G"], d["P"], d["G"]))
    x_work = np.array([d["P"]*d["G"] for d in stats_A], dtype=float)
    y_med  = np.array([d["runtime_median"] for d in stats_A], dtype=float)
    y_lo   = np.array([d["runtime_median"] - d["runtime_q25"] for d in stats_A], dtype=float)
    y_hi   = np.array([d["runtime_q75"] - d["runtime_median"] for d in stats_A], dtype=float)
    axA.errorbar(x_work, y_med, yerr=[y_lo, y_hi], fmt='o', capsize=3)
    good = np.isfinite(x_work) & np.isfinite(y_med)
    if good.sum() >= 2:
        coef = np.polyfit(x_work[good], y_med[good], deg=1)
        xs = np.linspace(x_work[good].min(), x_work[good].max(), 200)
        axA.plot(xs, np.polyval(coef, xs))
        yhat = np.polyval(coef, x_work[good])
        ss_res = np.sum((y_med[good] - yhat)**2)
        ss_tot = np.sum((y_med[good] - y_med[good].mean())**2)
        r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)
        axA.text(0.05, 0.95, f"R²={r2:.3f}", transform=axA.transAxes, ha="left", va="top")

    axA.set_xscale("log")
    axA.set_xlabel("Work (population × generations)")
    axA.set_ylabel("Runtime (s)")
    axA.set_title("A) Runtime vs Work (μ=0.2, meta=rf)")

    # (B) Runtime vs μ at fixed workload (P=50, G=10); one line per meta
    P_fix, G_fix = 50, 10
    for meta in META_SET:
        subset = [r for r in rows if r.P == P_fix and r.G == G_fix and r.meta == meta]
        stats_B = compute_group_stats(subset, key_fields=("mu",))
        stats_B = sorted(stats_B, key=lambda d: d["mu"])
        x_mu = [d["mu"] for d in stats_B]
        y_med = [d["runtime_median"] for d in stats_B]
        y_lo  = [d["runtime_median"] - d["runtime_q25"] for d in stats_B]
        y_hi  = [d["runtime_q75"] - d["runtime_median"] for d in stats_B]
        if len(x_mu) > 0:
            axB.errorbar(x_mu, y_med, yerr=[y_lo, y_hi], marker='o', label=meta)

    axB.set_xlabel("Mutation rate (μ)")
    axB.set_ylabel("Runtime (s)")
    axB.set_title("B) Runtime vs Mutation (P=50, G=10)")
    axB.legend(title="meta")

    # (C) Runtime vs meta at 3 workloads (μ=0.2); grouped dots
    x_positions = np.arange(len(WORKLOADS))
    width = 0.25
    offset_map = {'rf': -width, 'lr': 0.0, 'svm': width}

    for meta in META_SET:
        xs, ys, yerr_lo, yerr_hi = [], [], [], []
        for i, (label, (P, G)) in enumerate(WORKLOADS.items()):
            subset = [r for r in rows if r.P == P and r.G == G and r.mu == 0.2 and r.meta == meta]
            stats_C = compute_group_stats(subset, key_fields=tuple())
            if len(stats_C) == 1:
                s = stats_C[0]
                xs.append(x_positions[i] + offset_map[meta])
                ys.append(s["runtime_median"])
                yerr_lo.append(s["runtime_median"] - s["runtime_q25"])
                yerr_hi.append(s["runtime_q75"] - s["runtime_median"])
            else:
                xs.append(x_positions[i] + offset_map[meta]); ys.append(np.nan); yerr_lo.append(0.0); yerr_hi.append(0.0)
        axC.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt='o', label=meta)

    axC.set_xticks(x_positions, list(WORKLOADS.keys()))
    axC.set_xlabel("Workload")
    axC.set_ylabel("Runtime (s)")
    axC.set_title("C) Runtime vs Meta at 3 Workloads (μ=0.2)")
    axC.legend(title="meta", ncols=3, loc="upper left")

    plt.tight_layout()
    figpath = os.path.join(output_dir, "figure1_runtime_icde.png")
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

    # E1: Runtime vs Work (μ=0.2, meta='rf')
    log.info("E1: Runtime vs Work (μ=0.2, meta=rf) over %d×%d grid with %d repeats", len(POP_SET), len(GEN_SET), REPEATS)
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

    # E2: Mutation effect at fixed workload (P=50, G=10) with meta ∈ {rf, lr, svm}
    P_fix, G_fix = 50, 10
    log.info("E2: Runtime vs μ at (P=%d, G=%d) for meta in %s with %d repeats", P_fix, G_fix, META_SET, REPEATS)
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

    # E3: Meta-learner differences at three workloads (μ=0.2)
    log.info("E3: Runtime vs meta at 3 workloads (μ=0.2) with %d repeats", REPEATS)
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

    plot_figure_1(out_dir, results)

    txt_path = os.path.join(out_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write("GA Runtime Study (RAMSeS) — Summary\n")
        f.write(f"Timestamp: {dt.datetime.now().isoformat()}\n")
        f.write(f"Dataset: {DATASET_NAME} / entity={ENTITY}\n")
        f.write(f"Data file: {DATASET_FILE}\n")
        f.write(f"Anomalies injected (fixed): {ANOMALIES}\n")
        f.write(f"Repeats per config: {REPEATS}\n")
        f.write(f"Models dir: {TRAINED_MODELS_DIR}\n\n")

        # E1: fastest/slowest by median runtime
        subset_A = [r for r in results if r.mu == 0.2 and r.meta == 'rf']
        stats_A = compute_group_stats(subset_A, key_fields=("P","G"))
        stats_A = [s for s in stats_A if math.isfinite(s["runtime_median"])]
        if stats_A:
            stats_A_sorted = sorted(stats_A, key=lambda d: d["runtime_median"])
            best = stats_A_sorted[:5]
            worst = stats_A_sorted[-5:]
            f.write("E1 — Fastest 5 configs (by median runtime):\n")
            for b in best:
                f.write(f"  P={b['P']:>3}, G={b['G']:>4}, work={b['P']*b['G']:>5}, "
                        f"median={b['runtime_median']:.4f}s, IQR=[{b['runtime_q25']:.4f},{b['runtime_q75']:.4f}]\n")
            f.write("E1 — Slowest 5 configs (by median runtime):\n")
            for w in worst:
                f.write(f"  P={w['P']:>3}, G={w['G']:>4}, work={w['P']*w['G']:>5}, "
                        f"median={w['runtime_median']:.4f}s, IQR=[{w['runtime_q25']:.4f},{w['runtime_q75']:.4f}]\n")
            f.write("\n")

        # E2: per-meta medians at fixed workload
        P_fix, G_fix = 50, 10
        f.write(f"E2 — Medians at fixed workload (P={P_fix}, G={G_fix}) by meta and μ:\n")
        for meta in META_SET:
            subset = [r for r in results if r.P == P_fix and r.G == G_fix and r.meta == meta]
            stats_B = compute_group_stats(subset, key_fields=("mu",))
            stats_B = sorted(stats_B, key=lambda d: d["mu"])
            for s in stats_B:
                f.write(f"  meta={meta:>3}, mu={s['mu']:>4.2f} -> median={s['runtime_median']:.4f}s "
                        f"(IQR=[{s['runtime_q25']:.4f},{s['runtime_q75']:.4f}])\n")
        f.write("\n")

        # Failures
        failures = [r for r in results if not math.isfinite(r.runtime_s)]
        f.write(f"Failures: {len(failures)}\n")
        if failures:
            f.write("Example errors (up to 5):\n")
            for r in failures[:5]:
                f.write(f"  P={r.P}, G={r.G}, mu={r.mu}, meta={r.meta}, err={r.error[:200]}\n")

    log.info("Saved summary: %s", txt_path)
    log.info("Done. All artifacts in: %s", out_dir)

if __name__ == "__main__":
    main()
