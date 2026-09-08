import numpy as np
import copy
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from Metrics.metrics import (range_based_precision_recall_f1_auc, prauc, f1_score,
                             f1_soft_score, rank_key, vus_score, vus_window)
from Utils.model_selection_utils import evaluate_model, ScoringTimeout
from Utils.pipeline_spec import DEFAULT_DECISION_METRICS, metrics_required
from loguru import logger
import matplotlib.pyplot as plt
from Explainability import ir
from Model_Selection.Sensitivity_robustness import surrogate_fidelity

# Keys `summarize_results` adds beside the per-model entries. Everything that
# walks the summary skips these, so a new ranking cannot leak into a loop that
# expects model names.
RANKING_KEYS = ("ranked_by_f1", "ranked_by_pr_auc", "ranked_by_vus")


def add_noise_to_data(data, noise_level=0.1):
    """Add Gaussian noise to the data."""
    noise = noise_level * np.random.normal(size=data.shape)
    return data + noise


def monte_carlo_simulation(test_data, trained_models, model_names, dataset, entity, n_simulations=100,
                           noise_level=0.1, metrics=DEFAULT_DECISION_METRICS):
    """
    Perform Monte Carlo simulation to assess model robustness.

    Args:
        test_data: Original test data.
        trained_models: Dictionary of trained models.
        model_names: List of model names.
        dataset: Dataset name.
        entity: Entity name.
        n_simulations: Number of Monte Carlo simulations.
        noise_level: Level of noise to add to the data.

    Returns:
        A dictionary containing aggregated performance metrics for each model.
    """
    # Validation: Check if data is too small for Monte Carlo testing
    labels = test_data.entities[0].labels
    
    # Ensure labels are 2D
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)
    
    min_data_size = 50  # Minimum required data points for Monte Carlo
    data_size = labels.shape[1] if labels.ndim > 1 else labels.shape[0]
    
    if data_size < min_data_size:
        logger.warning(f"Monte Carlo simulation skipped: data size {data_size} < minimum {min_data_size}")
        return {}
    
    # Check if we have both classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning(f"Monte Carlo simulation skipped: only one class present in labels (unique values: {unique_labels})")
        return {}
    
    results = {model_name: {'f1_scores': [], 'pr_auc_scores': [], 'vus_scores': []}
               for model_name in model_names}
    # Noise does not change the series length, so one window covers every
    # simulation and keeps the VUS values on one scale.
    want_vus = 'vus' in metrics_required(metrics)
    vus_win = vus_window(test_data.entities[0].Y) if want_vus else None

    for sim in range(n_simulations):
        logger.info(f"Simulation {sim + 1}/{n_simulations}")
        noisy_data = copy.deepcopy(test_data)
        noisy_data.entities[0].Y = add_noise_to_data(noisy_data.entities[0].Y, noise_level)

        for model_name in model_names:
            model = trained_models.get(model_name)
            if not model or model_name not in results:
                continue
            try:
                evaluation = evaluate_model(noisy_data, model, model_name)
            except ScoringTimeout:
                results.pop(model_name, None)
                continue
            y_true = evaluation['anomaly_labels'].flatten()
            y_scores = evaluation['entity_scores'].flatten()
            _, _, f1, pr_auc, _ = range_based_precision_recall_f1_auc(y_true, y_scores)
            # f1, precision, recall, TP, TN, FP, FN = f1_score(y_scores, y_true)
            # pr_auc = prauc(y_true, y_scores)
            results[model_name]['f1_scores'].append(f1)
            results[model_name]['pr_auc_scores'].append(pr_auc)
            if want_vus:
                results[model_name]['vus_scores'].append(vus_score(y_scores, y_true, vus_win))

    return results


def run_monte_carlo_simulation(test_data, trained_models, model_names, dataset, entity, n_simulations=100,
                               noise_level=0.1, explain=False, metrics=DEFAULT_DECISION_METRICS):
    """Run the entire Monte Carlo simulation process."""
    # Run Monte Carlo simulation
    results = monte_carlo_simulation(test_data, trained_models, model_names, dataset, entity, n_simulations,
                                     noise_level, metrics=metrics)

    # Handle empty results (when data is too small or invalid)
    if not results:
        logger.warning("Monte Carlo simulation returned empty results")
        return [], [], []

    # Summarize results
    summary = summarize_results(results)

    # Handle empty summary
    if not summary or 'ranked_by_f1' not in summary or 'ranked_by_pr_auc' not in summary:
        logger.warning("Monte Carlo summary is empty or incomplete")
        return [], [], []

    # Print summary and rankings
    print("Summary of Monte Carlo Simulation:")
    for model_name, model_metrics in summary.items():
        if model_name not in RANKING_KEYS:
            print(f"Model: {model_name}")
            print(f"  F1 Mean: {model_metrics['f1_mean']:.4f}, F1 Std: {model_metrics['f1_std']:.4f}")
            print(f"  PR AUC Mean: {model_metrics['pr_auc_mean']:.4f}, "
                  f"PR AUC Std: {model_metrics['pr_auc_std']:.4f}")

    print("\nModels ranked by F1 score:")
    ranked_models_F1 = []
    for rank, model_name in enumerate(summary['ranked_by_f1'], 1):
        print(f"{rank}. {model_name}")
        ranked_models_F1.append(model_name)
    ranked_models_PR = []
    print("\nModels ranked by PR AUC score:")
    for rank, model_name in enumerate(summary['ranked_by_pr_auc'], 1):
        print(f"{rank}. {model_name}")
        ranked_models_PR.append(model_name)
    ranked_models_VUS = list(summary.get('ranked_by_vus') or [])
    if ranked_models_VUS:
        print("\nModels ranked by VUS score:")
        for rank, model_name in enumerate(ranked_models_VUS, 1):
            print(f"{rank}. {model_name}")

    # Save summary
    save_summary(summary, dataset, entity)

    # One F1/PR-AUC histogram pair per detector, no longer drawn. Eleven figures
    # per entity that say separately what the noise curves say together, and the
    # comparison between detectors is the whole point of this stage — which is
    # why the WebUI stopped listing them. plot_monte_carlo_results is kept below
    # so a single detector's distribution can still be minted by hand.
    # plot_monte_carlo_results(results, summary, model_names, dataset, entity)

    # Explainability (separate, explain-only noise sweep; production ranking above is unchanged)
    if explain:
        try:
            explain_monte_carlo(test_data, trained_models, model_names, dataset, entity,
                                explain=True, metrics=metrics,
                                production_rankings=(ranked_models_F1, ranked_models_PR,
                                                     ranked_models_VUS))
        except Exception as e:
            logger.error(f"Monte Carlo explainability failed (non-fatal): {e}")

    return ranked_models_F1, ranked_models_PR, ranked_models_VUS




def summarize_results(results):
    """Summarize the results of Monte Carlo simulation and rank models."""
    summary = {}
    for model_name, metrics in results.items():
        f1_mean = np.mean(metrics['f1_scores'])
        f1_std = np.std(metrics['f1_scores'])
        pr_auc_mean = np.mean(metrics['pr_auc_scores'])
        pr_auc_std = np.std(metrics['pr_auc_scores'])
        vus = [v for v in (metrics.get('vus_scores') or []) if not np.isnan(v)]
        summary[model_name] = {
            'f1_mean': f1_mean,
            'f1_std': f1_std,
            'pr_auc_mean': pr_auc_mean,
            'pr_auc_std': pr_auc_std,
            'vus_mean': float(np.mean(vus)) if vus else float('nan'),
            'vus_std': float(np.std(vus)) if vus else float('nan'),
        }

    has_vus = any(not np.isnan(s['vus_mean']) for s in summary.values())

    # Rank models by F1 score
    ranked_by_f1 = sorted(summary.items(), key=lambda x: x[1]['f1_mean'], reverse=True)
    ranked_by_pr_auc = sorted(summary.items(), key=lambda x: x[1]['pr_auc_mean'], reverse=True)
    ranked_by_vus = sorted(summary.items(), key=lambda x: rank_key(x[1]['vus_mean']), reverse=True)

    summary['ranked_by_f1'] = [item[0] for item in ranked_by_f1]
    summary['ranked_by_pr_auc'] = [item[0] for item in ranked_by_pr_auc]
    summary['ranked_by_vus'] = [item[0] for item in ranked_by_vus] if has_vus else []

    return summary


def plot_monte_carlo_results(results, summary, model_names, dataset, entity):
    """Plot the results of Monte Carlo simulation."""
    for model_name in model_names:
        f1_scores = results[model_name]['f1_scores']
        pr_auc_scores = results[model_name]['pr_auc_scores']

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(f1_scores, bins=20, alpha=0.7, color='blue')
        plt.title(f'{model_name} F1 Scores')
        plt.xlabel('F1 Score')
        plt.ylabel('Frequency')
        plt.axvline(summary[model_name]['f1_mean'], color='red', linestyle='dashed', linewidth=2)
        plt.text(summary[model_name]['f1_mean'] + summary[model_name]['f1_std'], max(plt.ylim()) * 0.9,
                 f'Mean: {summary[model_name]["f1_mean"]:.2f}\nStd: {summary[model_name]["f1_std"]:.2f}',
                 color='red')

        plt.subplot(1, 2, 2)
        plt.hist(pr_auc_scores, bins=20, alpha=0.7, color='green')
        plt.title(f'{model_name} PR AUC Scores')
        plt.xlabel('PR AUC Score')
        plt.ylabel('Frequency')
        plt.axvline(summary[model_name]['pr_auc_mean'], color='red', linestyle='dashed', linewidth=2)
        plt.text(summary[model_name]['pr_auc_mean'] + summary[model_name]['pr_auc_std'], max(plt.ylim()) * 0.9,
                 f'Mean: {summary[model_name]["pr_auc_mean"]:.2f}\nStd: {summary[model_name]["pr_auc_std"]:.2f}',
                 color='red')

        plt.tight_layout()

        # Save the plot
        directory = f'myresults/robustness/MonteCarlo/{dataset}/{entity}/'
        os.makedirs(directory, exist_ok=True)
        filename = f'{dataset}_{entity}_{model_name}_MonteCarloResults.png'
        plt.savefig(os.path.join(directory, filename), dpi=300)

        # plt.show()


def save_summary(summary, dataset, entity):
    """Save the summary of Monte Carlo simulation to a file."""
    directory = f'myresults/robustness/MonteCarlo/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    summary_file = os.path.join(directory, f'{dataset}_{entity}_MonteCarloSummary.txt')

    with open(summary_file, 'w') as f:
        f.write("Summary of Monte Carlo Simulation:\n")
        for model_name, model_metrics in summary.items():
            if model_name not in RANKING_KEYS:
                f.write(f"Model: {model_name}\n")
                f.write(f"  F1 Mean: {model_metrics['f1_mean']:.4f}, "
                        f"F1 Std: {model_metrics['f1_std']:.4f}\n")
                f.write(f"  PR AUC Mean: {model_metrics['pr_auc_mean']:.4f}, "
                        f"PR AUC Std: {model_metrics['pr_auc_std']:.4f}\n")
                if not np.isnan(model_metrics.get('vus_mean', float('nan'))):
                    f.write(f"  VUS Mean: {model_metrics['vus_mean']:.4f}, "
                            f"VUS Std: {model_metrics['vus_std']:.4f}\n")

        f.write("\nModels ranked by F1 score:\n")
        for rank, model_name in enumerate(summary['ranked_by_f1'], 1):
            f.write(f"{rank}. {model_name}\n")

        f.write("\nModels ranked by PR AUC score:\n")
        for rank, model_name in enumerate(summary['ranked_by_pr_auc'], 1):
            f.write(f"{rank}. {model_name}\n")

        if summary.get('ranked_by_vus'):
            f.write("\nModels ranked by VUS score:\n")
            for rank, model_name in enumerate(summary['ranked_by_vus'], 1):
                f.write(f"{rank}. {model_name}\n")


# ════════════════════════════════════════════════════════════════════════════
#  Monte Carlo Robustness Explainability
#
#  The production MC test holds noise_level fixed (0.1) — nothing structured
#  varies, so an explainer must exercise the test's one by-design knob. This
#  EXPLAIN-ONLY layer sweeps `noise_level` across a range (reusing the test's own
#  add_noise_to_data), records per-trial (noise_level → per-model F1/PR-AUC), and
#  explains the result two complementary ways, both 1-D over noise_level:
#    (A) performance-vs-noise curves (crossovers, win-regions, breakdown points,
#        ranking stability),
#    (B) a 1-D decision-tree surrogate (noise→winner threshold rules + per-model
#        degradation regressors).
#  The production ranking (fixed noise) is untouched.
# ════════════════════════════════════════════════════════════════════════════

# 21 points, not 20: `linspace(0, 0.2, 20)` steps by 0.0105 and lands on values
# like 0.0421 and 0.1789, so every axis label and every reported breakdown point
# was an artefact of the point count. 21 gives an exact 0.01 step — 0.00, 0.01,
# … 0.20 — which is readable and quotable.
DEFAULT_NOISE_LEVELS = np.linspace(0.0, 0.2, 21)


def _mc_data_feasible(test_data) -> bool:
    """Same infeasibility guards as monte_carlo_simulation (data <50 pts / single-class)."""
    labels = test_data.entities[0].labels
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)
    data_size = labels.shape[1] if labels.ndim > 1 else labels.shape[0]
    if data_size < 50:
        return False
    if len(np.unique(labels)) < 2:
        return False
    return True


def _best_f1_and_cut(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """Best point-wise F1 over the data-driven quantile thresholds of y_scores, and
    the threshold (score cut) that achieves it. Returns (best_f1, best_cut); best_cut
    is NaN if no threshold yields F1 > 0 (e.g. a degenerate/constant score)."""
    best_f1, best_cut = 0.0, float('nan')
    for _t in np.unique(np.quantile(y_scores, np.linspace(0.0, 1.0, 21))):
        _f1 = f1_score((y_scores >= _t).astype(int), y_true)[0]
        if _f1 > best_f1:
            best_f1 = float(_f1); best_cut = float(_t)
    return best_f1, best_cut


def _f1_at_cut(y_true: np.ndarray, y_scores: np.ndarray, cut: float) -> float:
    """Point-wise F1 at a FIXED score threshold `cut` (no re-optimization). NaN cut → NaN."""
    if cut is None or np.isnan(cut):
        return float('nan')
    return float(f1_score((y_scores >= cut).astype(int), y_true)[0])


def monte_carlo_noise_sweep(
    test_data, trained_models, model_names,
    noise_levels=None, repeats: int = 5, random_state: int = 0,
    evaluate_fn: Optional[Callable[[str, float], Tuple[float, float]]] = None,
    metrics=DEFAULT_DECISION_METRICS,
) -> Optional[Dict[str, Any]]:
    """
    Explain-only sweep over the MC test's own `noise_level`. For each level × each
    repeat: add Gaussian noise (via add_noise_to_data) to a deep copy, evaluate
    every model, record the row.

    Scoring uses FAST point-wise metrics (PR-AUC + best-F1 over data-driven quantile
    thresholds of each trial's own score distribution), NOT the slow range-based
    metric the production MC uses: this sweep does hundreds of evaluations, and the
    range-based windowing metric (~30 s/call on long series) would make it take many
    hours. The production ranking path is untouched.

    Two F1 variants are recorded per trial (PR-AUC is threshold-free, so it has only one):
      • F1       — ADAPTIVE best threshold, re-optimized at every noise level (measures
                   best-achievable separability vs noise).
      • F1_fixed — FIXED operating point: each model's best threshold is chosen once at
                   the LOWEST noise level and held constant across the sweep (measures how
                   a committed decision threshold degrades as noise shifts the scores).
    The grid is sorted ascending so its first level is the baseline where thresholds freeze.

    Only the metrics the run's fitness names are computed; the others come back as
    NaN columns, so the arrays stay rectangular and nothing downstream has to branch.

    evaluate_fn(model_name, level) -> (f1, pr[, f1_fixed[, vus]]) is injectable for
    tests; when given, real noising/evaluation is skipped (a 2-tuple sets
    f1_fixed = f1, and vus = NaN). Per-model failures → NaN.

    Returns {noise (n=L*R,), grid_levels (L,), F1 (n×M), F1_fixed (n×M), PR (n×M),
    VUS (n×M), metrics, model_names}, or None when the data is too small /
    single-class.
    """
    if not _mc_data_feasible(test_data):
        return None
    grid = np.sort(np.asarray(DEFAULT_NOISE_LEVELS if noise_levels is None else noise_levels, dtype=float))
    rng = np.random.RandomState(random_state)

    want = metrics_required(metrics)
    want_f1, want_pr, want_vus = 'f1' in want, 'pr_auc' in want, 'vus' in want
    vus_win = vus_window(test_data.entities[0].Y) if want_vus else None
    noise_col: List[float] = []
    F1: List[List[float]] = []
    F1_fixed: List[List[float]] = []
    PR: List[List[float]] = []
    VUS: List[List[float]] = []
    # Per-model threshold frozen at the baseline (lowest) noise level.
    baseline_cuts: Dict[str, float] = {}
    for i, level in enumerate(grid):
        is_baseline = (i == 0)
        logger.info(f"MC explain sweep: noise {level:.3f} ({i + 1}/{len(grid)}), {repeats} repeats")
        for _ in range(repeats):
            noisy = None
            if evaluate_fn is None:
                np.random.seed(int(rng.randint(0, 2 ** 31 - 1)))
                noisy = copy.deepcopy(test_data)
                noisy.entities[0].Y = add_noise_to_data(noisy.entities[0].Y, float(level))
            row_f1: List[float] = []
            row_f1_fixed: List[float] = []
            row_pr: List[float] = []
            row_vus: List[float] = []
            for m in model_names:
                try:
                    if evaluate_fn is not None:
                        vals = evaluate_fn(m, float(level))
                        f1v, prv = float(vals[0]), float(vals[1])
                        f1_fixedv = float(vals[2]) if len(vals) > 2 else f1v
                        vusv = float(vals[3]) if len(vals) > 3 else float('nan')
                    else:
                        model = trained_models.get(m)
                        if not model:
                            row_f1.append(float('nan')); row_f1_fixed.append(float('nan'))
                            row_pr.append(float('nan')); row_vus.append(float('nan')); continue
                        ev = evaluate_model(noisy, model, m)
                        y_true = ev['anomaly_labels'].flatten()
                        y_scores = np.asarray(ev['entity_scores'].flatten(), dtype=float)
                        # Fast point-wise scoring (see docstring): PR-AUC + best-F1 over
                        # DATA-DRIVEN quantile thresholds of THIS trial's own score
                        # distribution (not a fixed 0.1–0.9 grid — detectors emit
                        # un-normalised scores on different scales, and a fixed grid
                        # would miss any whose scores saturate high (NN) or low).
                        if want_pr:
                            try:
                                prv = float(prauc(y_true, y_scores))
                            except Exception:
                                prv = float('nan')
                        else:
                            prv = float('nan')
                        if want_f1:
                            f1v, cut = _best_f1_and_cut(y_true, y_scores)
                            # Freeze each model's best threshold at the baseline level,
                            # then reuse it at every level for the fixed-operating-point F1.
                            if is_baseline and m not in baseline_cuts:
                                baseline_cuts[m] = cut
                            f1_fixedv = _f1_at_cut(y_true, y_scores,
                                                   baseline_cuts.get(m, float('nan')))
                        else:
                            f1v = f1_fixedv = float('nan')
                        vusv = (vus_score(y_scores, y_true, vus_win) if want_vus
                                else float('nan'))
                    row_f1.append(float(f1v)); row_f1_fixed.append(float(f1_fixedv))
                    row_pr.append(float(prv)); row_vus.append(float(vusv))
                except Exception as e:
                    logger.error(f"MC explain sweep: model {m} failed at noise {level}: {e}")
                    row_f1.append(float('nan')); row_f1_fixed.append(float('nan'))
                    row_pr.append(float('nan')); row_vus.append(float('nan'))
            noise_col.append(float(level)); F1.append(row_f1)
            F1_fixed.append(row_f1_fixed); PR.append(row_pr); VUS.append(row_vus)

    return {
        "noise": np.asarray(noise_col, dtype=float),
        "grid_levels": grid,
        "F1": np.asarray(F1, dtype=float),
        "F1_fixed": np.asarray(F1_fixed, dtype=float),
        "PR": np.asarray(PR, dtype=float),
        "VUS": np.asarray(VUS, dtype=float),
        "metrics": want,
        "model_names": list(model_names),
    }


# ── Method A: curves, crossover, breakdown, ranking stability (pure) ─────────

def compute_noise_curves(noise, grid_levels, score_matrix, model_names,
                         breakdown_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Per noise level: per-model mean+std score, the top model, the crossover points
    where the top model changes, per-model win-regions (contiguous noise intervals
    where the model leads), and per-model breakdown point (smallest level whose mean
    score < breakdown_threshold; None if it never breaks down).
    """
    grid = np.asarray(grid_levels, dtype=float)
    L = len(grid)
    M = len(model_names)
    per_model_mean = np.full((M, L), np.nan)
    per_model_std = np.full((M, L), np.nan)
    for li, lvl in enumerate(grid):
        mask = np.isclose(noise, lvl)
        if not np.any(mask):
            continue
        sub = score_matrix[mask]
        with np.errstate(invalid='ignore'):
            per_model_mean[:, li] = np.nanmean(sub, axis=0)
            per_model_std[:, li] = np.nanstd(sub, axis=0)

    winner_per_level: List[Optional[str]] = []
    for li in range(L):
        col = per_model_mean[:, li]
        winner_per_level.append(None if np.all(np.isnan(col))
                                else model_names[int(np.nanargmax(col))])

    crossovers: List[Dict[str, Any]] = []
    for li in range(1, L):
        a, b = winner_per_level[li - 1], winner_per_level[li]
        if a is not None and b is not None and a != b:
            crossovers.append({"noise": float(grid[li]), "from_model": a, "to_model": b})

    win_regions: Dict[str, List[Tuple[float, float]]] = {m: [] for m in model_names}
    li = 0
    while li < L:
        w = winner_per_level[li]
        if w is None:
            li += 1; continue
        lj = li
        while lj + 1 < L and winner_per_level[lj + 1] == w:
            lj += 1
        win_regions[w].append((float(grid[li]), float(grid[lj])))
        li = lj + 1

    breakdown: Dict[str, Optional[float]] = {}
    for mi, m in enumerate(model_names):
        bp = None
        for li in range(L):
            v = per_model_mean[mi, li]
            if not np.isnan(v) and v < breakdown_threshold:
                bp = float(grid[li]); break
        breakdown[m] = bp

    return {
        "grid_levels": grid,
        "per_model_mean": per_model_mean,
        "per_model_std": per_model_std,
        "winner_per_level": winner_per_level,
        "crossovers": crossovers,
        "win_regions": win_regions,
        "breakdown_points": breakdown,
    }


def _ranks_from_means(means: np.ndarray) -> np.ndarray:
    """Rank vector (0 = best) from a per-model mean-score vector; NaN treated as worst."""
    order = np.argsort(-np.nan_to_num(means, nan=-np.inf))
    ranks = np.empty(len(means), dtype=float)
    for pos, idx in enumerate(order):
        ranks[idx] = pos
    return ranks


def compute_ranking_stability(noise, grid_levels, score_matrix, model_names) -> Dict[str, Any]:
    """
    Per level: rank models by mean score and compute Kendall-τ vs the global
    aggregate ranking (over all trials). τ≈1 = stable ranking; lower = volatile.
    """
    from scipy.stats import kendalltau
    grid = np.asarray(grid_levels, dtype=float)
    with np.errstate(invalid='ignore'):
        global_rank = _ranks_from_means(np.nanmean(score_matrix, axis=0))
    taus: List[float] = []
    for lvl in grid:
        mask = np.isclose(noise, lvl)
        if not np.any(mask):
            taus.append(float('nan')); continue
        with np.errstate(invalid='ignore'):
            lvl_rank = _ranks_from_means(np.nanmean(score_matrix[mask], axis=0))
        tau, _ = kendalltau(global_rank, lvl_rank)
        taus.append(float(tau) if tau is not None and not np.isnan(tau) else float('nan'))
    return {"grid_levels": grid, "tau_per_level": np.asarray(taus, dtype=float)}


# ── Method B: 1-D decision-tree surrogate (lazy sklearn) ─────────────────────

def _fit_noise_winner(noise, score_matrix, model_names, max_depth: int = 3, random_state: int = 0):
    """Fit DecisionTreeClassifier(noise → argmax-model). Returns (clf_or_None, info)."""
    from collections import Counter
    noise = np.asarray(noise, dtype=float)
    rows: List[float] = []
    winners: List[str] = []
    for i in range(score_matrix.shape[0]):
        r = score_matrix[i]
        if np.all(np.isnan(r)):
            continue
        rows.append(noise[i]); winners.append(model_names[int(np.nanargmax(r))])
    if not rows:
        return None, {"feasible": False, "rules_text": "", "win_rates": {},
                      "train_accuracy": float('nan'), "classes": [], "root_threshold": None}
    cnt = Counter(winners)
    n = len(winners)
    win_rates = {m: cnt.get(m, 0) / n for m in model_names}
    classes = sorted(cnt)
    if len(classes) == 1:
        return None, {"feasible": True,
                      "rules_text": f"Always {classes[0]} (single winner across the sweep).",
                      "win_rates": win_rates, "train_accuracy": 1.0,
                      "cv_accuracy": 1.0, "cv_accuracy_std": 0.0, "cv_method": "n/a (single class)",
                      "classes": classes, "root_threshold": None}
    from sklearn.tree import DecisionTreeClassifier, export_text
    X = np.asarray(rows, dtype=float).reshape(-1, 1)
    y = np.asarray(winners)
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X, y)
    acc = float(clf.score(X, y))
    # In-sample accuracy (above) is what the exported tree/rules were fit to
    # reproduce; it is not, by itself, evidence that the tree generalizes
    # rather than having fit noise in this particular sweep. Report a
    # cross-validated estimate alongside it (see surrogate_fidelity.py).
    cv = surrogate_fidelity.held_out_classifier_fidelity(
        X, y, max_depth=max_depth, random_state=random_state)
    rules = export_text(clf, feature_names=["noise_level"])
    root_thr = float(clf.tree_.threshold[0]) if clf.tree_.node_count > 1 else None
    # Structured (machine-readable) rules for the IR layer; non-fatal if the
    # Explainability package is unavailable in a stripped-down environment.
    try:
        rules_structured = ir.tree_to_rules(clf, ["noise_level"])
    except Exception:
        rules_structured = []
    return clf, {"feasible": True, "rules_text": rules, "rules": rules_structured,
                 "win_rates": win_rates,
                 "train_accuracy": acc,
                 "cv_accuracy": cv["cv_accuracy"], "cv_accuracy_std": cv["cv_accuracy_std"],
                 "cv_method": cv["method"], "cv_note": cv["note"],
                 "classes": classes, "root_threshold": root_thr}


def train_noise_winner_surrogate(noise, score_matrix, model_names,
                                 max_depth: int = 3, random_state: int = 0) -> Dict[str, Any]:
    """Public wrapper returning just the winner-surrogate info dict (see _fit_noise_winner)."""
    _, info = _fit_noise_winner(noise, score_matrix, model_names, max_depth, random_state)
    return info


def train_noise_permodel_surrogates(noise, score_matrix, model_names,
                                    max_depth: int = 3, random_state: int = 0) -> Dict[str, Any]:
    """
    Per model: DecisionTreeRegressor(noise → score). Returns per model the in-sample
    R² (fit quality on the swept points), a cross-validated R² (held-out
    generalization estimate — see surrogate_fidelity.py / Molnar 2022), the
    trend ('robust' if Pearson corr of noise vs score ≥ 0 else 'fragile'), and
    the regressor's predicted score at the lowest and highest swept noise.
    """
    from sklearn.tree import DecisionTreeRegressor
    noise = np.asarray(noise, dtype=float)
    lo = float(np.min(noise)); hi = float(np.max(noise))
    out: Dict[str, Any] = {}
    for mi, m in enumerate(model_names):
        col = score_matrix[:, mi]
        mask = ~np.isnan(col)
        if int(mask.sum()) < 2:
            out[m] = {"trend": "N/A", "corr": float('nan'), "score_low": float('nan'),
                      "score_high": float('nan'), "r2": float('nan'),
                      "cv_r2": float('nan'), "cv_method": "n/a",
                      "cv_n_splits": 0, "cv_degenerate_folds": 0}
            continue
        X = noise[mask].reshape(-1, 1)
        ys = col[mask]
        reg = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        reg.fit(X, ys)
        r2 = float(reg.score(X, ys))
        cv = surrogate_fidelity.held_out_regressor_fidelity(
            X, ys, max_depth=max_depth, random_state=random_state)
        if np.std(noise[mask]) > 0 and np.std(ys) > 0:
            corr = float(np.corrcoef(noise[mask], ys)[0, 1])
        else:
            corr = 0.0
        out[m] = {"trend": "robust" if corr >= 0 else "fragile", "corr": corr,
                  "score_low": float(reg.predict([[lo]])[0]),
                  "score_high": float(reg.predict([[hi]])[0]), "r2": r2,
                  "cv_r2": cv.get("cv_r2", float('nan')), "cv_method": cv.get("method", "n/a"),
                  "cv_mse": cv.get("cv_mse", float('nan')),
                  "cv_n_splits": cv.get("n_splits", 0),
                  "cv_degenerate_folds": cv.get("n_degenerate_folds", 0)}
    return out


# ── Plots ────────────────────────────────────────────────────────────────────

def _mc_explain_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    })


def plot_noise_curves(curves, model_names, metric_name, dataset, entity, plain: bool = False) -> None:
    """Per-model score vs noise_level.

    plain=False (default): mean±std band + win-region shading + crossover markers +
        breakdown markers → ..._noise_curves_{tag}.png.
    plain=True: only the per-model mean lines (no bands/shading/markers) →
        ..._noise_curves_{tag}_plain.png.
    """
    _mc_explain_rcparams()
    grid = curves["grid_levels"]
    mean = curves["per_model_mean"]
    std = curves["per_model_std"]
    colour_map = {m: plt.cm.tab20(i / max(len(model_names), 1)) for i, m in enumerate(model_names)}

    fig, ax = plt.subplots(figsize=(10, 5))
    for mi, m in enumerate(model_names):
        ax.plot(grid, mean[mi], label=m, color=colour_map[m], linewidth=1.4)
        if not plain:
            ax.fill_between(grid, mean[mi] - std[mi], mean[mi] + std[mi],
                            color=colour_map[m], alpha=0.12, lw=0)

    if not plain:
        # Win-region shading, crossover markers, and breakdown markers.
        for m, regions in curves["win_regions"].items():
            for (a, b) in regions:
                ax.axvspan(a, b, color=colour_map.get(m, "#cccccc"), alpha=0.06, lw=0)
        for cx in curves["crossovers"]:
            ax.axvline(cx["noise"], color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        for m, bp in curves["breakdown_points"].items():
            if bp is not None:
                ax.scatter([bp], [0.5], marker="v", color=colour_map.get(m, "#888888"),
                           s=25, zorder=5)

    ax.set_xlabel("noise_level (Gaussian std)")
    ax.set_ylabel(f"{metric_name}" + ("" if plain else " (mean ± std over repeats)"))
    if plain:
        ax.set_title(f"Monte Carlo · {metric_name} vs noise level (per-model means)")
    else:
        ax.set_title(f"Monte Carlo · {metric_name} vs noise level "
                     "(shaded = win-region; ▼ = breakdown)")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout(pad=1.2)
    directory = f"myresults/robustness/MonteCarlo/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    tag = metric_name.replace("-", "").replace(" ", "")
    suffix = "_plain" if plain else ""
    plt.savefig(f"{directory}/{dataset}_{entity}_MonteCarlo_noise_curves_{tag}{suffix}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_ranking_stability(stab_f1, stab_pr, dataset, entity, stab_vus=None) -> None:
    """Kendall-τ of each noise level's ranking vs the aggregate ranking.

    One line per metric the run swept; a metric it did not sweep passes None.
    """
    _mc_explain_rcparams()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for stab, label, marker, colour in (
            (stab_f1, "F1 ranking", "o", "#1f77b4"),
            (stab_pr, "PR-AUC ranking", "s", "#2ca02c"),
            (stab_vus, "VUS ranking", "^", "#d62728")):
        if stab:
            ax.plot(stab["grid_levels"], stab["tau_per_level"], marker=marker,
                    markersize=3, label=label, color=colour, linewidth=1.4)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("noise_level (Gaussian std)")
    ax.set_ylabel("Kendall τ vs aggregate ranking")
    ax.set_ylim(-1.05, 1.08)
    ax.set_title("Monte Carlo · ranking stability vs noise level")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="lower left", frameon=False)
    plt.tight_layout(pad=1.2)
    directory = f"myresults/robustness/MonteCarlo/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/{dataset}_{entity}_MonteCarlo_ranking_stability.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_surrogate_tree(clf, metric_name, dataset, entity) -> None:
    """plot_tree of the small noise→winner classifier. No-op when clf is None."""
    if clf is None:
        return
    from sklearn.tree import plot_tree
    _mc_explain_rcparams()
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_tree(clf, feature_names=["noise_level"], class_names=list(clf.classes_),
              filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title(f"Monte Carlo · winner surrogate ({metric_name}): noise_level → winning model")
    plt.tight_layout(pad=1.2)
    directory = f"myresults/robustness/MonteCarlo/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    tag = metric_name.replace("-", "").replace(" ", "")
    plt.savefig(f"{directory}/{dataset}_{entity}_MonteCarlo_surrogate_tree_{tag}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


# ── Orchestrator + report ────────────────────────────────────────────────────

def explain_monte_carlo(test_data, trained_models, model_names, dataset, entity,
                        noise_levels=None, repeats: int = 5, random_state: int = 0,
                        explain: bool = False,
                        evaluate_fn: Optional[Callable[[str, float], Tuple[float, float]]] = None,
                        production_rankings: Optional[Tuple[List[str], ...]] = None,
                        metrics=DEFAULT_DECISION_METRICS,
                        ) -> Optional[Dict[str, Any]]:
    """
    Monte Carlo robustness explainability: sweep the test's `noise_level`, then
    explain with performance-vs-noise curves + ranking stability (Method A) and a
    1-D decision-tree surrogate (Method B), for each metric the run's fitness
    names. Writes a report + plots under
    myresults/robustness/MonteCarlo/{dataset}/{entity}/.

    Returns the computed structures when explain=True; None otherwise (and None,
    with a logged note, when the sweep is infeasible).
    """
    if not explain:
        return None
    sweep = monte_carlo_noise_sweep(test_data, trained_models, model_names,
                                    noise_levels=noise_levels, repeats=repeats,
                                    random_state=random_state, evaluate_fn=evaluate_fn,
                                    metrics=metrics)
    if sweep is None:
        logger.warning("Monte Carlo explainability skipped: data too small / single-class.")
        return None

    noise = sweep["noise"]; grid = sweep["grid_levels"]
    F1 = sweep["F1"]; PR = sweep["PR"]; models = sweep["model_names"]
    F1_fixed = sweep["F1_fixed"]; VUS = sweep["VUS"]
    want = metrics_required(metrics)
    want_f1, want_pr, want_vus = 'f1' in want, 'pr_auc' in want, 'vus' in want

    curves_f1 = compute_noise_curves(noise, grid, F1, models) if want_f1 else {}
    curves_pr = compute_noise_curves(noise, grid, PR, models) if want_pr else {}
    curves_vus = compute_noise_curves(noise, grid, VUS, models) if want_vus else {}
    curves_f1_fixed = compute_noise_curves(noise, grid, F1_fixed, models) if want_f1 else {}
    stab_f1 = compute_ranking_stability(noise, grid, F1, models) if want_f1 else {}
    stab_pr = compute_ranking_stability(noise, grid, PR, models) if want_pr else {}
    stab_vus = compute_ranking_stability(noise, grid, VUS, models) if want_vus else {}

    # Surrogates (need sklearn). Degrade gracefully if unavailable.
    clf_f1 = clf_pr = clf_vus = None
    winner_f1 = winner_pr = winner_vus = {"feasible": False}
    permodel_f1 = permodel_pr = permodel_vus = {}
    surrogate_note = ""
    try:
        if want_f1:
            clf_f1, winner_f1 = _fit_noise_winner(noise, F1, models)
            permodel_f1 = train_noise_permodel_surrogates(noise, F1, models)
        if want_pr:
            clf_pr, winner_pr = _fit_noise_winner(noise, PR, models)
            permodel_pr = train_noise_permodel_surrogates(noise, PR, models)
        if want_vus:
            clf_vus, winner_vus = _fit_noise_winner(noise, VUS, models)
            permodel_vus = train_noise_permodel_surrogates(noise, VUS, models)
    except ImportError:
        surrogate_note = "scikit-learn unavailable — surrogate (Method B) skipped."
        logger.warning(f"MC explainability: {surrogate_note}")

    # Plots, one set per swept metric.
    for curves, label in ((curves_f1, "F1"), (curves_pr, "PR-AUC"),
                          (curves_vus, "VUS"), (curves_f1_fixed, "F1_fixed")):
        if curves:
            plot_noise_curves(curves, models, label, dataset, entity)
            plot_noise_curves(curves, models, label, dataset, entity, plain=True)
    plot_ranking_stability(stab_f1, stab_pr, dataset, entity, stab_vus=stab_vus)
    for clf, label in ((clf_f1, "F1"), (clf_pr, "PR-AUC"), (clf_vus, "VUS")):
        if clf is not None:
            plot_surrogate_tree(clf, label, dataset, entity)

    directory = f"myresults/robustness/MonteCarlo/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    report_path = os.path.join(directory, f"{dataset}_{entity}_MonteCarlo_explainability.txt")
    n_trials = len(noise)
    with open(report_path, "w") as f:
        f.write("=== Monte Carlo Robustness Explainability ===\n")
        f.write(f"Dataset: {dataset}  |  Entity: {entity}\n")
        f.write(f"Models ({len(models)}): {', '.join(models)}\n")
        f.write(f"Noise sweep: {len(grid)} levels in "
                f"[{grid.min():.3f}, {grid.max():.3f}] × {repeats} repeats = {n_trials} trials\n")
        f.write("(Explain-only sweep over the test's own noise_level; the production MC "
                "ranking at fixed noise is unchanged.)\n\n")

        def _methodA(curves, stab, metric):
            f.write(f"--- Method A · {metric} curves ---\n")
            f.write("Crossovers (noise → new top model):\n")
            if curves["crossovers"]:
                for cx in curves["crossovers"]:
                    f.write(f"    noise {cx['noise']:.3f}: {cx['from_model']} → {cx['to_model']}\n")
            else:
                f.write("    none (one model leads across the whole sweep)\n")
            f.write("Per-model win-region in noise_level:\n")
            for m in models:
                regs = curves["win_regions"][m]
                if regs:
                    rs = ", ".join(f"[{a:.3f},{b:.3f}]" for a, b in regs)
                    f.write(f"    {m:<12} {rs}\n")
            f.write("Per-model breakdown noise (mean score first < 0.5):\n")
            for m in models:
                bp = curves["breakdown_points"][m]
                f.write(f"    {m:<12} {'never' if bp is None else f'{bp:.3f}'}\n")
            taus = stab["tau_per_level"]
            mean_tau = float(np.nanmean(taus)) if np.any(~np.isnan(taus)) else float('nan')
            if np.any(~np.isnan(taus)):
                worst = int(np.nanargmin(taus))
                worst_band = f"{stab['grid_levels'][worst]:.3f} (τ={taus[worst]:+.3f})"
            else:
                worst_band = "N/A"
            f.write(f"Ranking stability: mean τ={mean_tau:+.3f}; most volatile at noise {worst_band}\n\n")

        for _curves, _stab, _label in ((curves_f1, stab_f1, "F1"),
                                       (curves_pr, stab_pr, "PR-AUC"),
                                       (curves_vus, stab_vus, "VUS")):
            if _curves:
                _methodA(_curves, _stab, _label)

        def _methodB(winner, permodel, metric):
            f.write(f"--- Method B · winner surrogate ({metric}) ---\n")
            if surrogate_note:
                f.write(f"    {surrogate_note}\n\n"); return
            if not winner.get("feasible"):
                f.write("    not feasible (no valid trials).\n\n"); return
            f.write(f"Train accuracy (in-sample fit): {winner['train_accuracy']:.3f}\n")
            cv_acc = winner.get("cv_accuracy", float('nan'))
            if not np.isnan(cv_acc):
                f.write(f"Held-out accuracy ({winner.get('cv_method', 'cv')}, "
                        f"{winner.get('cv_accuracy_std', float('nan')):.3f} std): {cv_acc:.3f}\n")
            elif winner.get("cv_note"):
                f.write(f"Held-out accuracy: not estimated ({winner['cv_note']})\n")
            wr = winner["win_rates"]
            top = sorted(wr.items(), key=lambda kv: kv[1], reverse=True)
            f.write("Win rates: " + ", ".join(f"{m} {p:.2f}" for m, p in top if p > 0) + "\n")
            f.write("Threshold rules (noise_level → winner):\n")
            for line in winner["rules_text"].rstrip().splitlines():
                f.write(f"    {line}\n")
            f.write(f"\n--- Method B · per-model degradation ({metric}) ---\n")
            f.write(f"      {'model':<12} {'trend':>8} {'score@low':>10} {'score@high':>11} "
                    f"{'R² (train)':>11} {'R² (held-out)':>22}\n")
            f.write("      " + "-" * 65 + "\n")
            for m in models:
                pm = permodel.get(m, {})
                tr = pm.get("trend", "N/A")
                sl = pm.get("score_low", float('nan'))
                sh = pm.get("score_high", float('nan'))
                r2 = pm.get("r2", float('nan'))
                cv_r2 = pm.get("cv_r2", float('nan'))
                cv_mse = pm.get("cv_mse", float('nan'))
                cv_ns = pm.get("cv_n_splits", 0)
                cv_deg = pm.get("cv_degenerate_folds", 0)
                if not np.isnan(cv_r2):
                    # Majority-degenerate folds → the estimate is unassessable;
                    # keep the number visible but flag it (matches the IR grade).
                    if cv_ns and cv_deg > cv_ns / 2:
                        cv_str = f"{cv_r2:.3f} ({cv_deg}/{cv_ns} deg,N/A)"
                    elif cv_deg:
                        cv_str = f"{cv_r2:.3f} ({cv_deg}/{cv_ns} deg)"
                    else:
                        cv_str = f"{cv_r2:.3f}"
                elif not np.isnan(cv_mse):
                    cv_str = f"MSE={cv_mse:.3f}"
                elif pm.get("cv_method") == "constant_target":
                    cv_str = "N/A (flat)"
                else:
                    cv_str = "N/A"
                f.write(f"      {m:<12} {tr:>8} "
                        f"{(f'{sl:.3f}' if not np.isnan(sl) else 'N/A'):>10} "
                        f"{(f'{sh:.3f}' if not np.isnan(sh) else 'N/A'):>11} "
                        f"{(f'{r2:.3f}' if not np.isnan(r2) else 'N/A'):>11} "
                        f"{cv_str:>22}\n")
            f.write("      R² (train) is the in-sample fit; R² (held-out) is a cross-validated\n"
                    "      estimate (see surrogate_fidelity.py) and is the number that should be\n"
                    "      read as the surrogate's actual fidelity. '(k/n deg)' marks folds with\n"
                    "      near-constant targets (force_finite-scored); majority-degenerate\n"
                    "      estimates are flagged N/A — kept visible but not to be trusted.\n")
            f.write("\n")

        for _winner, _permodel, _label, _token in (
                (winner_f1, permodel_f1, "F1", "f1"),
                (winner_pr, permodel_pr, "PR-AUC", "pr_auc"),
                (winner_vus, permodel_vus, "VUS", "vus")):
            if _token in want:
                _methodB(_winner, _permodel, _label)

        # ── Fixed-operating-point degradation ──────────────────────────────
        # The F1 above (Methods A/B) RE-OPTIMIZES each model's threshold at every
        # noise level (best-achievable separability). Here the threshold is frozen
        # once at the lowest noise level and held fixed across the sweep — how a
        # committed operating point actually degrades as noise shifts the scores.
        if want_f1:
            f.write("--- Fixed-operating-point degradation (F1, threshold frozen at lowest noise) ---\n")
            f.write("Contrast with the adaptive F1 above (threshold re-optimized at every level).\n")
            f.write(f"      {'model':<12} {'f1@low':>8} {'f1@high(adapt)':>15} "
                    f"{'f1@high(fixed)':>15} {'masked drop':>12}\n")
            f.write("      " + "-" * 64 + "\n")
            ad_mean = curves_f1["per_model_mean"]
            fx_mean = curves_f1_fixed["per_model_mean"]

            def _fnum(v):
                return f"{v:.3f}" if not np.isnan(v) else "N/A"

            for mi, m in enumerate(models):
                f_low = ad_mean[mi, 0]
                f_hi_ad = ad_mean[mi, -1]
                f_hi_fx = fx_mean[mi, -1]
                masked = (f_hi_ad - f_hi_fx) if not (np.isnan(f_hi_ad) or np.isnan(f_hi_fx)) else float('nan')
                f.write(f"      {m:<12} {_fnum(f_low):>8} {_fnum(f_hi_ad):>15} "
                        f"{_fnum(f_hi_fx):>15} {_fnum(masked):>12}\n")
            f.write("'masked drop' = adaptive F1 − fixed F1 at the highest noise: the "
                    "operating-point degradation the adaptive (re-optimized) view hides.\n\n")

        f.write("Note: Method A relates the noise level to which model leads (curves, "
                "crossovers, win-regions, breakdown, ranking stability). Method B is a 1-D "
                "decision tree — it formalizes the curve crossovers as explicit noise "
                "thresholds and quantifies each model's degradation. The fixed-operating-point "
                "section instead freezes each threshold at baseline to expose committed-cutoff "
                "robustness.\n")

    result = {
        "sweep": sweep,
        "curves_f1": curves_f1, "curves_pr": curves_pr, "curves_vus": curves_vus,
        "curves_f1_fixed": curves_f1_fixed,
        "stability_f1": stab_f1, "stability_pr": stab_pr, "stability_vus": stab_vus,
        "winner_f1": winner_f1, "winner_pr": winner_pr, "winner_vus": winner_vus,
        "permodel_f1": permodel_f1, "permodel_pr": permodel_pr,
        "permodel_vus": permodel_vus,
        "metrics": want,
        "n_trials": n_trials,
    }

    # ── Intermediate Representation (grounded LLM input; non-fatal) ─────────
    try:
        ranks = list(production_rankings or ()) + [None, None, None]
        ir.write_stage_ir(
            ir.build_monte_carlo_ir(
                dataset, entity, result,
                ranks[0] if want_f1 else None,
                ranks[1] if want_pr else None,
                ranks[2] if want_vus else None),
            dataset, entity, "ir_monte_carlo")
    except Exception as e:
        logger.error(f"Monte Carlo IR emission failed (non-fatal): {e}")

    return result