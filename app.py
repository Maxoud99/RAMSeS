# time_series_framework/app.py
import concurrent.futures
import copy
import logging
import os
import psutil
import sys
import threading
import time
import traceback
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch as t

from Datasets.load import load_data, provides_real_labels
from Datasets import tsb_ad
from Metrics.Ensemble_GA import (
    genetic_algorithm,
    evaluate_individual_models,
    fitness_function,
)
from Model_Selection.Sensitivity_robustness.GAN_test import run_Gan
from Model_Selection.Sensitivity_robustness.Monte_Carlo_Simulation import (
    run_monte_carlo_simulation,
)
from Model_Selection.Sensitivity_robustness.off_by_threshold_testing import (
    run_off_by_threshold,
)
from Model_Selection.Thompson_Sampling import run_linear_thompson_sampling
from Model_Selection.inject_anomalies import Inject
from Model_Selection.rank_aggregation import (
    enhanced_markov_chain_rank_aggregator_text,
    explain_rank_aggregation,
)
from Model_Training.train import TrainModels, TRAINING_TIMEOUT_SECONDS
from Metrics.metrics import vus_score, vus_window
from Utils.model_io import load_detector
from Utils.model_selection_utils import (
    describe_failure,
    probe_detector,
    timed_out_detectors,
)
from Utils.pipeline_spec import (
    ALL_DETECTORS,
    combine_metrics,
    count_windows,
    online_feasibility,
    online_split,
    ONLINE_LABELS_REAL,
    ONLINE_LABEL_SOURCE_LABELS,
    resolve_online_labels,
    REGIME_SHIFT_TYPES,
    dataset_label,
    DEFAULT_ANOMALY_TYPE,
    DEFAULT_DECISION_METRICS,
    decision_metric_formula,
    decision_metric_label,
    metrics_required,
    ranking_metrics_for,
    restrict_metrics,
    DETECTOR_FAMILIES,
    MIN_DETECTORS,
    families_for,
    family_of,
    MULTIVARIATE_FAMILIES,
    UNIVARIATE_FAMILIES,
)
from Utils.pipeline_spec import ALL_STAGES as _SPEC_ALL_STAGES
from Utils.pipeline_spec import OFFLINE_ITERATION as _SPEC_OFFLINE_ITERATION
from Utils.utils import get_args_from_cmdline
from Utils.paths import results_dir
from Utils.set_all_seeds import set_all_seeds
# from comprehensive_results_writer import write_comprehensive_results

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# save_dir removed - now dynamically determined from command-line args
# Old hardcoded path was causing wrong models to load for different datasets

# The detector/stage vocabulary lives in Utils/pipeline_spec.py so that app.py,
# Utils/utils.py (the CLI) and the web UI all read one definition.
algorithm_list = list(DETECTOR_FAMILIES)
algorithm_list_instances = list(ALL_DETECTORS)

# Sub-stages of the offline model-selection pipeline (stage 6). The --stages CLI
# flag selects a subset; a strict subset is a "partial run" (runs only those
# stages + their explainability, then stops before rank aggregation).
ALL_STAGES = set(_SPEC_ALL_STAGES)

# Iteration tag of the OFFLINE model-selection run. The offline pipeline and its
# artifacts (rank-aggregation reports, explanation IRs, narration) all use this
# value; the CLI --iteration arg is the window-sizing parameter, a different
# concept that must not leak into these filenames.
OFFLINE_ITERATION = _SPEC_OFFLINE_ITERATION

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def get_memory_usage_mb():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

def get_peak_memory_mb():
    """High-water RSS of this process in MB, or None if no peak is obtainable.

    `resource` is Unix-only. This used to fall back through a bare `except:` to
    `get_memory_usage_mb()` — a CURRENT reading, printed under the heading
    "Peak Memory" with nothing to say it was not one. On Windows that made the
    reported peak silently meaningless. The fallback is now the sampler's own
    high-water mark, which is a real peak (sampled rather than exact); None
    only when neither source can answer, and the report says so rather than
    printing a number.
    """
    try:
        import resource
    except ImportError:
        return _MEM_TRACE.peak_all()
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes, Linux in kilobytes.
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


# Processes that serve the local LLM. The model runs OUTSIDE this interpreter —
# `narrate_entity` reaches it over HTTP — so `psutil.Process()` never sees the
# 9 GB qwen2.5:14b holds, and the LLM stage used to be reported as ~20 MB.
# Matched by name because the server is a plain OS process with no handle we
# own; the runner subprocess that actually holds the weights matches too.
_LLM_PROCESS_HINTS = ("ollama",)


def _fmt_metric(value, digits=6):
    """A metric that could not be computed reads as 'not available', not 0."""
    try:
        if value is None or np.isnan(float(value)):
            return "not available"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "not available"


def llm_server_memory_mb():
    """Summed RSS of the local LLM server processes, or None if none are up.

    None rather than 0.0 on purpose: "the server was not running" and "the
    server used no memory" are different claims, and the report says so.
    """
    total = None
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmd = " ".join(proc.info.get("cmdline") or ()).lower()
            if any(h in name or h in cmd for h in _LLM_PROCESS_HINTS):
                total = (total or 0.0) + proc.memory_info().rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return total


class _MemoryTrace:
    """Background sampler of this process's RSS and the LLM server's.

    A module's memory is a PEAK over its window, not the difference of two
    endpoint samples. RSS at an instant is what the OS chose to keep resident,
    so endpoint deltas go negative whenever pages are reclaimed — which is
    exactly what a 9 GB model loading in another process causes. The delta is
    still reported (it answers "what did this stage leave behind"), but the
    peak is what answers "how much did this stage need".
    """

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._samples = []          # (t, self_rss_mb, llm_rss_mb or None)
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self._samples.append(
                    (time.time(), get_memory_usage_mb(), llm_server_memory_mb()))
            except Exception:
                pass                # sampling must never break a run

    def start(self):
        if self._thread is None:
            self._samples.append(
                (time.time(), get_memory_usage_mb(), llm_server_memory_mb()))
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="memory-trace")
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval)
            self._thread = None

    def _window(self, t0, t1):
        return [s for s in self._samples if t0 <= s[0] <= t1]

    def peak_between(self, t0, t1):
        """Highest RSS of THIS process seen in [t0, t1]. None if unsampled."""
        vals = [s[1] for s in self._window(t0, t1)]
        return max(vals) if vals else None

    def peak_all(self):
        """Highest RSS of this process across the whole trace. None if unsampled."""
        return max((x[1] for x in self._samples), default=None)

    def llm_peak_between(self, t0, t1):
        """Highest summed RSS of the LLM server in [t0, t1]. None if it was down."""
        vals = [s[2] for s in self._window(t0, t1) if s[2] is not None]
        return max(vals) if vals else None


# One trace for the whole run; started in main() and read per module.
_MEM_TRACE = _MemoryTrace()


def write_comprehensive_results(output_file, dataset, entity, iteration, results_dict, timing_dict, memory_dict=None):
    """
    Write comprehensive results including timing, scores, and chosen models.
    
    Parameters
    ----------
    output_file : str
        Path to output file
    dataset : str
        Dataset name
    entity : str
        Entity identifier
    iteration : int
        Iteration number
    results_dict : dict
        Dictionary containing all results
    timing_dict : dict
        Dictionary containing timing information
    memory_dict : dict, optional
        Dictionary containing memory usage information
        Dictionary containing timing information
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"RAMSeS Framework - Comprehensive Results\n")
        f.write(f"Dataset: {dataset_label(dataset)} | Entity: {entity} | Iteration: {iteration}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # ============ COMPUTATIONAL OVERHEAD ============
        f.write("="*80 + "\n")
        f.write("COMPUTATIONAL OVERHEAD (seconds)\n")
        f.write("="*80 + "\n\n")
        
        f.write("Per-Module Timing:\n")
        f.write("-" * 50 + "\n")
        total_module_time = 0
        for module, elapsed in timing_dict.get('modules', {}).items():
            f.write(f"  {module:<30} : {elapsed:>10.4f}s\n")
            total_module_time += elapsed
        
        f.write("-" * 50 + "\n")
        f.write(f"  {'Total Module Time':<30} : {total_module_time:>10.4f}s\n")
        f.write(f"  {'End-to-End Time':<30} : {timing_dict.get('total', 0):>10.4f}s\n")
        f.write(f"  {'Overhead (E2E - Modules)':<30} : {timing_dict.get('total', 0) - total_module_time:>10.4f}s\n\n")
        
        # ============ MEMORY FOOTPRINT ============
        if memory_dict:
            f.write("="*80 + "\n")
            f.write("MEMORY FOOTPRINT (MB)\n")
            f.write("="*80 + "\n\n")
            
            f.write("Per-Module Memory Usage:\n")
            f.write("  Peak is the highest resident size sampled while the module ran —\n")
            f.write("  what it needed. Delta is end minus start — what it left behind, and\n")
            f.write("  negative whenever the OS reclaimed pages during the module.\n")
            f.write("-" * 50 + "\n")
            shared_peak = False
            for module, mem_info in memory_dict.get('modules', {}).items():
                f.write(f"  {module:<30} : {mem_info['after']:>10.2f} MB\n")
                peak = mem_info.get('peak')
                if peak is not None:
                    mark = " (shared)" if mem_info.get('peak_shared') else ""
                    shared_peak = shared_peak or bool(mem_info.get('peak_shared'))
                    f.write(f"    Peak : {peak:>10.2f} MB{mark}\n")
                f.write(f"    Delta: {mem_info['delta']:>+10.2f} MB\n")
                llm_peak = mem_info.get('llm_server_peak')
                if llm_peak is not None:
                    f.write(f"    LLM server peak (separate process) "
                            f": {llm_peak:>10.2f} MB\n")
                elif 'llm_server_peak' in mem_info:
                    f.write("    LLM server peak (separate process) "
                            ":  not measured (server not running)\n")
            if shared_peak:
                f.write("  (shared) = stages ran concurrently in one process, so their\n")
                f.write("            peaks are not separable; this is the parallel region's.\n")
            
            f.write("-" * 50 + "\n")
            initial = memory_dict.get('initial', 0)
            final = memory_dict.get('final', 0)
            peak = memory_dict.get('peak')
            f.write(f"  {'Initial Memory':<30} : {initial:>10.2f} MB\n")
            if peak is not None:
                f.write(f"  {'Peak Memory':<30} : {peak:>10.2f} MB\n")
                f.write(f"  {'Peak above start':<30} : {peak - initial:>+10.2f} MB\n")
            else:
                f.write(f"  {'Peak Memory':<30} :   unavailable on this platform\n")
            final_alg = memory_dict.get('final_algorithms')
            if final_alg is not None:
                f.write(f"  {'Final Memory (algorithms)':<30} : {final_alg:>10.2f} MB\n")
                f.write(f"  {'Final Memory (after LLM)':<30} : {final:>10.2f} MB\n")
            else:
                f.write(f"  {'Final Memory':<30} : {final:>10.2f} MB\n")
            f.write(f"  {'Total Increase (final-start)':<30} : {final - initial:>+10.2f} MB\n")
            llm_init = memory_dict.get('llm_server_initial')
            llm_peaks = [m.get('llm_server_peak') for m in memory_dict.get('modules', {}).values()
                         if m.get('llm_server_peak') is not None]
            if llm_peaks:
                f.write(f"  {'LLM server peak':<30} : {max(llm_peaks):>10.2f} MB"
                        f"  (separate process, not in the figures above)\n")
                if llm_init is not None:
                    f.write(f"  {'LLM server at start':<30} : {llm_init:>10.2f} MB\n")
            f.write("\n")
        
        # ============ GENETIC ALGORITHM (ENSEMBLE) ============
        f.write("="*80 + "\n")
        f.write("GENETIC ALGORITHM - ENSEMBLE SELECTION\n")
        f.write("="*80 + "\n\n")
        
        ga_results = results_dict.get('ga', {})
        f.write(f"Best Ensemble: {ga_results.get('ensemble', 'N/A')}\n")
        f.write(f"  F1 Score    : {ga_results.get('f1', 0):.6f}\n")
        f.write(f"  PR-AUC      : {ga_results.get('pr_auc', 0):.6f}\n")
        f.write(f"  Fitness     : {ga_results.get('fitness', 0):.6f}\n")
        f.write(f"  Meta-Model  : {ga_results.get('meta_model_type', 'N/A')}\n")
        f.write(f"  Ensemble Size: {len(ga_results.get('ensemble', []))}\n\n")
        
        # ============ THOMPSON SAMPLING ============
        f.write("="*80 + "\n")
        f.write("THOMPSON SAMPLING - ONLINE MODEL SELECTION\n")
        f.write("="*80 + "\n\n")
        
        thompson_results = results_dict.get('thompson', {})
        f.write(f"Chosen Model: {thompson_results.get('best_model', 'N/A')}\n")
        f.write(f"  F1 Score    : {thompson_results.get('f1', 0):.6f}\n")
        f.write(f"  PR-AUC      : {thompson_results.get('pr_auc', 0):.6f}\n\n")
        
        f.write("Top-5 Models (Ranked):\n")
        for i, model in enumerate(thompson_results.get('top_models', [])[:5], 1):
            f.write(f"  {i}. {model}\n")
        f.write("\n")
        
        # ============ ROBUSTNESS TESTS ============
        f.write("="*80 + "\n")
        f.write("ROBUSTNESS TESTS - INDIVIDUAL RANKINGS\n")
        f.write("="*80 + "\n\n")
        
        # GAN Results
        f.write("GAN Robustness Test:\n")
        f.write("-" * 50 + "\n")
        gan_results_rob = results_dict.get('gan_robustness', {})
        f.write(f"Chosen Model: {gan_results_rob.get('best_model', 'N/A')}\n")
        f.write(f"  F1 Score    : {gan_results_rob.get('best_f1', 0):.6f}\n")
        f.write(f"  PR-AUC      : {gan_results_rob.get('best_pr_auc', 0):.6f}\n\n")
        
        f.write("  Top-5 by F1:\n")
        for i, model in enumerate(gan_results_rob.get('f1_names', [])[:5], 1):
            f.write(f"    {i}. {model}\n")
        f.write("  Top-5 by PR-AUC:\n")
        for i, model in enumerate(gan_results_rob.get('pr_auc_names', [])[:5], 1):
            f.write(f"    {i}. {model}\n")
        f.write("\n")
        
        # Borderline/Off-by-Threshold Results
        f.write("Borderline Sensitivity Test:\n")
        f.write("-" * 50 + "\n")
        borderline_results = results_dict.get('borderline', {})
        f.write(f"Chosen Model: {borderline_results.get('best_model', 'N/A')}\n")
        f.write(f"  F1 Score    : {borderline_results.get('best_f1', 0):.6f}\n")
        f.write(f"  PR-AUC      : {borderline_results.get('best_pr_auc', 0):.6f}\n\n")
        
        f.write("  Top-5 by F1:\n")
        for i, model in enumerate(borderline_results.get('f1_names', [])[:5], 1):
            f.write(f"    {i}. {model}\n")
        f.write("  Top-5 by PR-AUC:\n")
        for i, model in enumerate(borderline_results.get('pr_auc_names', [])[:5], 1):
            f.write(f"    {i}. {model}\n")
        f.write("\n")
        
        # Monte Carlo Results
        f.write("Monte Carlo Simulation:\n")
        f.write("-" * 50 + "\n")
        mc_results = results_dict.get('monte_carlo', {})
        f.write(f"Chosen Model (F1)   : {mc_results.get('best_model_f1', 'N/A')}\n")
        f.write(f"Chosen Model (PR-AUC): {mc_results.get('best_model_pr_auc', 'N/A')}\n")
        f.write(f"  Best F1 Score      : {mc_results.get('best_f1', 0):.6f}\n")
        f.write(f"  Best PR-AUC Score  : {mc_results.get('best_pr_auc', 0):.6f}\n\n")
        
        f.write("  Top-5 by F1:\n")
        for i, model in enumerate(mc_results.get('f1_names', [])[:5], 1):
            f.write(f"    {i}. {model}\n")
        f.write("  Top-5 by PR-AUC:\n")
        for i, model in enumerate(mc_results.get('pr_auc_names', [])[:5], 1):
            f.write(f"    {i}. {model}\n")
        f.write("\n")
        
        # ============ RANK AGGREGATION ============
        f.write("="*80 + "\n")
        f.write("RANK AGGREGATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        aggregation_results = results_dict.get('aggregation', {})
        
        f.write("Robust Aggregation (GAN → Borderline → Monte Carlo):\n")
        f.write("-" * 50 + "\n")
        robust_agg = aggregation_results.get('robust_agg', [])
        if isinstance(robust_agg, (list, tuple)) and len(robust_agg) > 0:
            f.write(f"  Best Model: {robust_agg[0]}\n")
            f.write(f"  Full Ranking: {robust_agg}\n")
        else:
            f.write(f"  Result: {robust_agg}\n")
        f.write("\n")

        f.write("Final Aggregation (Robust + Thompson Sampling):\n")
        f.write("-" * 50 + "\n")
        final_agg = aggregation_results.get('final_agg', [])
        if isinstance(final_agg, (list, tuple)) and len(final_agg) > 0:
            f.write(f"  Best Model: {final_agg[0]}\n")
            f.write(f"  Full Ranking: {final_agg}\n")
        else:
            f.write(f"  Result: {final_agg}\n")
        f.write("\n")
        
        # ============ FRAMEWORK FINAL DECISION ============
        f.write("="*80 + "\n")
        f.write("FRAMEWORK FINAL DECISION\n")
        f.write("="*80 + "\n\n")
        
        final_decision = results_dict.get('final_decision', {})
        
        # Show both options
        f.write("Single Model Option:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Model       : {final_decision.get('single_model', 'N/A')}\n")
        f.write(f"  F1 Score    : {final_decision.get('single_model_f1', 0):.6f}\n")
        f.write(f"  PR-AUC      : {final_decision.get('single_model_pr_auc', 0):.6f}\n")
        f.write(f"  VUS         : {_fmt_metric(final_decision.get('single_model_vus'))}\n\n")
        
        f.write("Ensemble Option:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Models      : {final_decision.get('ensemble', 'N/A')}\n")
        f.write(f"  Size        : {len(final_decision.get('ensemble', []))}\n")
        f.write(f"  Meta-Model  : {final_decision.get('meta_model_type', 'N/A')}\n")
        f.write(f"  F1 Score    : {final_decision.get('ensemble_f1', 0):.6f}\n")
        f.write(f"  PR-AUC      : {final_decision.get('ensemble_pr_auc', 0):.6f}\n")
        f.write(f"  VUS         : {_fmt_metric(final_decision.get('ensemble_vus'))}\n")
        f.write(f"  Fitness     : {final_decision.get('ensemble_fitness', 0):.6f}\n\n")
        
        # Reads the decision rather than re-deriving it, so the report cannot
        # disagree with the choice the pipeline actually made.
        metric = final_decision.get('decision_metric', ('f1',))
        label = decision_metric_label(metric)
        formula = decision_metric_formula(metric)
        ensemble_score = final_decision.get('ensemble_score',
                                            final_decision.get('ensemble_f1', 0))
        single_score = final_decision.get('single_model_score',
                                          final_decision.get('single_model_f1', 0))

        f.write("Final Choice:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Fitness    : {formula}\n")
        if final_decision.get('framework_choice') == 'ensemble':
            f.write(f"  ✓ ENSEMBLE SELECTED\n")
            f.write(f"    Reason: Ensemble fitness ({formula}, {ensemble_score:.6f}) >= Single Model fitness ({single_score:.6f})\n")
            if ensemble_score > single_score:
                improvement = ensemble_score - single_score
                f.write(f"    Improvement: +{improvement:.6f} ({improvement/max(single_score, 0.0001)*100:.2f}%)\n")
            else:
                f.write(f"    Note: fitness scores are equal\n")
        else:
            f.write(f"  ✓ SINGLE MODEL SELECTED\n")
            f.write(f"    Reason: Single Model fitness ({formula}, {single_score:.6f}) > Ensemble fitness ({ensemble_score:.6f})\n")
            advantage = single_score - ensemble_score
            f.write(f"    Advantage: +{advantage:.6f} ({advantage/max(ensemble_score, 0.0001)*100:.2f}%)\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    logger.info(f"Comprehensive results saved to: {output_file}")


def load_trained_models(model_names, models_dir, probe_data=None):
    """
    Load trained models from disk, keeping only those that can score the data.

    Parameters
    ----------
    model_names : list[str]
        List of model instance names (e.g., 'CBLOF_1', 'LOF_3', ...).
    models_dir : str
        Directory where .pth files are stored.
    probe_data : Dataset, optional
        The data this run will score. When given, every checkpoint that loads
        is also probed against it, and one that cannot score it is excluded
        rather than returned.

    Returns
    -------
    (dict[str, torch.nn.Module], dict[str, str])
        The usable models, and {name: reason} for every checkpoint that exists
        but was excluded — because it could not be loaded at all, or because
        the probe found it cannot score `probe_data`. A name in neither had no
        checkpoint.
    """
    trained = {}
    missing = []
    excluded = {}

    def exclude(name, failure, reason):
        excluded[name] = reason
        # Worded for WebUI/markers.py, which turns each of these lines into its
        # own banner: the name, the reason and the remedy all have to be on
        # this one line, and both kinds of exclusion share the prefix so one
        # pattern there catches them.
        logger.warning(
            f"⚠ Incompatible checkpoint {name} excluded: it {failure} — "
            f"{reason.rstrip('.')}. Remedy: retrain it with --overwrite true. "
            f"The checkpoint file was left untouched.")

    for name in model_names:
        path = os.path.join(models_dir, f"{name}.pth")
        if not os.path.exists(path):
            missing.append(name)
            logger.warning(f"Model {name} not found in {models_dir}, skipping")
            continue
        try:
            with open(path, 'rb') as fh:
                # One call owns both halves of reading a checkpoint back: the
                # pickler the save used, and putting the weights AND every
                # device the model recorded on one device. The repair that used
                # to sit here ran only when CUDA was absent, so on a GPU host a
                # GPU-trained detector kept `device='cuda'` over CPU weights and
                # crashed on its first forward pass; Utils/model_io.py has the
                # full account.
                model = load_detector(fh)
        except Exception as e:
            # A checkpoint that cannot be read at all is the stalest kind
            # there is, and the store has two sorts of it: pickles that name
            # the `algorithm` package this code was once laid out as (whole
            # entities of the smd and anomaly_archive trees, every family), and
            # files cut short by a save that died partway, since
            # Logger.save_torch_model writes the .pth in place. With
            # --overwrite false TrainModels takes any .pth on disk as trained,
            # so this is the last place to catch either before it ends the run
            # with a traceback that names no detector and throws away every
            # model that did load. Any Exception, because what unpickling can
            # raise is open-ended — ModuleNotFoundError, AttributeError for a
            # renamed class, torch's zip reader's RuntimeError, EOFError — and
            # every one of them means the same thing for this run.
            exclude(name, "cannot be loaded", describe_failure(e))
            continue
        if probe_data is not None:
            # A checkpoint that unpickles can still be unable to score this
            # data — written by an older version of its detector, or for a
            # dataset with another channel count — and that used to surface as
            # a traceback from inside the GA that ended the whole run. Probing
            # here turns it into one exclusion with a reason, before any stage
            # starts. Nothing is retrained, deleted or rewritten: a checkpoint
            # is the user's to replace.
            reason = probe_detector(probe_data, model, name)
            if reason is not None:
                if name in timed_out_detectors():
                    # A probe that ran out of time has already been reported by
                    # the deadline itself, and retraining is not its remedy.
                    excluded[name] = reason
                else:
                    exclude(name, "cannot score this data", reason)
                continue
        trained[name] = model

    logger.info("Loaded %d trained models from %s", len(trained), models_dir)
    if missing:
        logger.warning(f"Missing {len(missing)} models: {', '.join(missing)}")
    return trained, excluded

# ------------------------------------------------------------------------------
# Model-Selection Pipelines
# ------------------------------------------------------------------------------

# The leading phrase of the warning a failed GAN stage logs. WebUI/markers.py
# matches this exact text to list it among the run's warnings, so change the
# two together (test_app.py checks they still agree).
GAN_FAILED_WARNING = "GAN robustness test failed"


def _run_gan_or_skip(*args, **kwargs):
    """`run_Gan`, failing soft: returns None where it would have raised.

    The GAN test supplies one of the three robustness rankings. If it is lost,
    the consensus is built from the other two. If it raises, run_app's
    catch-all marks the whole entity as failed, and Off-by, Monte Carlo, rank
    aggregation, the final decision and the report are lost with it. The known
    trigger is TensorFlow's GPU stack (see Utils/tf_device.py), but any failure
    here is handled the same way: one warning, then the run continues as it
    would with --skip_gan.

    Only Exception is caught. A KeyboardInterrupt still stops the run, and a
    native abort inside TensorFlow cannot be caught by Python at all. Keeping
    TensorFlow off the GPU is the defence against that.
    """
    try:
        return run_Gan(*args, **kwargs)
    except Exception as e:
        # First line only: TensorFlow's errors run to dozens of lines, and the
        # warning should stay a single log line. The full text and traceback
        # follow it through exc_info.
        detail = str(e).strip().splitlines()
        logger.warning(
            f"⚠ {GAN_FAILED_WARNING} and was skipped "
            f"({type(e).__name__}: {detail[0] if detail else 'no message'}). "
            f"The remaining robustness tests and the rank aggregation run "
            f"without it, as with --skip_gan.",
            exc_info=True)
        return None


def _retire_gan_explanation(dataset, entity):
    """Delete the GAN explanation an earlier run left for this entity.

    Call it when an --explain run that covers the GAN stage ends up with no GAN
    result, because the stage failed soft or --skip_gan was set. A successful
    stage overwrites ir_gan.json and nl_gan.txt. A failed or skipped one writes
    neither, so the earlier run's pair stays on disk. Three readers then take
    it for this run's explanation: the global IR lists it as an "ok" stage
    summary, the narrator rewrites its prose, and the result page shows the
    GAN card next to the warning that the test failed.

    Only these two files go. The plots under robustness/GAN are left in place,
    because the result page shows them only on a GAN card, and without these
    files there is no card. A partial run that leaves GAN out of --stages does
    not call this. It keeps the earlier GAN files, as it does for every stage
    it did not run, and it assembles no global IR that could claim them as its
    own.
    """
    for path in (
            os.path.join(results_dir("explanations_ir"), str(dataset), str(entity), "ir_gan.json"),
            os.path.join(results_dir("explanations_nl"), str(dataset), str(entity), "nl_gan.txt")):
        try:
            os.remove(path)
        except FileNotFoundError:
            continue
        except OSError as e:
            # Not fatal. The global IR still marks the stage not available,
            # because it reads this run's empty GAN ranking and ignores the
            # file (see Explainability.ir.assemble_global_ir).
            logger.warning(f"Could not remove the earlier run's GAN explanation "
                           f"{path} ({e}); the result page may still show it.")
            continue
        logger.info(f"     Removed the GAN explanation from an earlier run "
                    f"({path}): this run has no GAN result to explain.")


def run_model_selection_algorithms_1(train_data, test_data, dataset, entity, iteration, model_list=None, test_data_gan=None, skip_gan=False, explain=False, stages=None, decision_metric=DEFAULT_DECISION_METRICS, real_labels=False):
    """
    One-pass model selection pipeline in the order:
      1) GA (stacking ensemble search)
      2) Thompson Sampling (LinTS)
      3) GAN robustness test (optional, can be skipped for speed)
      4) Off-by-threshold (borderline sensitivity)
      5) Monte Carlo (noise stress test)
      6) Rank aggregations (robust-only, then merged with Thompson)

    Parameters
    ----------
    model_list : list[str], optional
        List of model names to use. If None, uses global algorithm_list_instances.
    test_data_gan : Dataset, optional
        Separate dataset for GAN test (should be full original data without injection).
        GAN creates its own perturbations internally. If None, uses test_data.
    skip_gan : bool, optional
        If True, skips GAN test for faster re-optimization in online phase.
    real_labels : bool, optional
        `test_data` carries REAL labels and nothing injected (a real-label
        re-optimisation of the online phase). Thompson Sampling then skips
        windows whose labels are one class instead of rewarding a detector
        for a metric that is undefined there. False everywhere else.

    Returns (exactly 11 items):
        thompson_model_names[0],
        robust_agg[1],
        full_aggregated[1],
        best_ensemble,
        individual_predictions,
        base_model_predictions_train,
        base_model_predictions_test,
        y_true_train,
        y_true_test,
        meta_model_type,
        extra_results_dict
    """
    # Use provided model list or fall back to global
    models_to_use = model_list if model_list is not None else algorithm_list_instances

    def still_usable():
        """The pool minus detectors killed for slowness by an earlier stage."""
        slow = timed_out_detectors()
        return [m for m in models_to_use if m not in slow]

    # Which sub-stages to run (a strict subset of ALL_STAGES is a "partial run":
    # those stages execute, then the function returns before rank aggregation).
    stages = set(ALL_STAGES) if stages is None else set(stages)

    timing_dict = {}
    memory_dict = {'modules': {}}

    # Safe defaults so the return tuple is well-formed even when a stage is skipped.
    best_ensemble = []
    best_f1 = best_pr_auc = best_fitness = 0.0
    individual_predictions = None
    base_model_predictions_train = base_model_predictions_test = None
    y_true_train = y_true_test = None
    meta_model_type = None
    ensemble_scores = None
    # The GA's fitted meta-model of best_ensemble; the online phase deploys it.
    best_meta_model = None
    thompson_model_names = []
    Gan_ranked_by_f1 = Gan_ranked_by_pr_auc = []
    Gan_ranked_by_f1_names = Gan_ranked_by_pr_auc_names = Gan_ranked_by_vus_names = []
    ranked_by_f1 = ranked_by_pr_auc = []
    ranked_by_f1_names_sensitivity = ranked_by_pr_auc_names_sensitivity = []
    ranked_by_vus_names_sensitivity = []
    monte_carlo_ranked_models_F1 = monte_carlo_ranked_models_PR = []
    monte_carlo_ranked_models_VUS = []
    robust_agg = [None, []]
    full_aggregated = [None, []]

    # Track initial memory. The sampler runs alongside from here, so every
    # module gets a PEAK over its own window rather than two endpoint samples.
    _MEM_TRACE.start()
    initial_memory = get_memory_usage_mb()
    memory_dict['initial'] = initial_memory
    memory_dict['llm_server_initial'] = llm_server_memory_mb()
    logger.info(f"  💾 Initial memory usage: {initial_memory:.2f} MB")

    # -------------------------
    # 1) Genetic Algorithm (GA)
    # -------------------------
    if "ga" in stages:
        logger.info("  📊 Sub-stage 6.1: Genetic Algorithm (GA) - Finding best ensemble...")
        logger.info("     This will evaluate individual models and run 20 generations")
        mem_before = get_memory_usage_mb()
        start_time = time.time()
        best_ensemble, best_f1, best_pr_auc, best_fitness, \
        individual_predictions, base_model_predictions_train, base_model_predictions_test, \
        y_true_train, y_true_test, meta_model_type, ensemble_scores, best_meta_model = genetic_algorithm(
            dataset, entity, train_data, test_data,
            still_usable(), trained_models,
            population_size=20, generations=20,
            meta_model_type='rf', mutation_rate=0.1,
            explain=explain,
            metric=decision_metric,
            vus_win=vus_window(test_data.entities[0].Y),
            return_meta_model=True,
        )
        timing_dict['1_Genetic_Algorithm'] = time.time() - start_time
        mem_after = get_memory_usage_mb()
        memory_dict['modules']['1_Genetic_Algorithm'] = {
            'before': mem_before,
            'after': mem_after,
            'delta': mem_after - mem_before,
            'peak': _MEM_TRACE.peak_between(start_time, time.time()),
        }
        logger.info(
            "  ✓ [GA] Best ensemble=%s | F1=%.4f | PR-AUC=%.4f | fitness=%.4f | Time=%.4fs",
            best_ensemble, best_f1, best_pr_auc, best_fitness, timing_dict['1_Genetic_Algorithm']
        )

    # -----------------------------------
    # 2) Thompson Sampling (LinTS, online)
    # -----------------------------------
    if "thompson" in stages:
        logger.info("  📊 Sub-stage 6.2: Thompson Sampling - Online model selection...")
        mem_before = get_memory_usage_mb()
        start_time = time.time()
        thompson_model_names = run_linear_thompson_sampling(
            test_data=test_data,
            trained_models=trained_models,
            model_names=still_usable(),
            dataset=dataset,
            entity=entity,
            iterations=50,
            iteration=iteration,
            explain=explain,
            metrics=decision_metric,
            skip_one_class_windows=real_labels,
        )
        timing_dict['2_Thompson_Sampling'] = time.time() - start_time
        mem_after = get_memory_usage_mb()
        memory_dict['modules']['2_Thompson_Sampling'] = {
            'before': mem_before,
            'after': mem_after,
            'delta': mem_after - mem_before,
            'peak': _MEM_TRACE.peak_between(start_time, time.time()),
        }
        logger.info("  ✓ [Thompson] Top-5: %s | Time=%.4fs", thompson_model_names[:5],
                    timing_dict['2_Thompson_Sampling'])

    # -------------------------
    # 3) GAN Robustness Testing
    # -------------------------
    if "gan" in stages and not skip_gan:
        logger.info("  📊 Sub-stage 6.3: GAN Robustness Testing...")
        mem_before = get_memory_usage_mb()
        start_time = time.time()
        # GAN uses separate dataset (full original data) to avoid circularity
        # GAN creates its own perturbations internally
        test_data_for_gan = copy.deepcopy(test_data_gan if test_data_gan is not None else test_data)
        gan_results = _run_gan_or_skip(
            test_data_for_gan, trained_models, still_usable(), dataset, entity,
            explain=explain, metrics=decision_metric
        )
        # Time and memory are recorded on failure too: the stage did run for
        # that long, and the end-to-end total includes it.
        timing_dict['3_GAN_Robustness'] = time.time() - start_time
        mem_after = get_memory_usage_mb()
        memory_dict['modules']['3_GAN_Robustness'] = {
            'before': mem_before,
            'after': mem_after,
            'delta': mem_after - mem_before,
            'peak': _MEM_TRACE.peak_between(start_time, time.time()),
        }
        if gan_results is None:
            # Failed soft: the GAN rankings keep their empty defaults from
            # above, exactly as under --skip_gan. The SKIPPED marker moves the
            # web UI's stage rail off "running".
            logger.info("  ⏩ Sub-stage 6.3: GAN Robustness Testing SKIPPED "
                        "(the stage failed; see the warning above)")
            if explain:
                _retire_gan_explanation(dataset, entity)
        else:
            Gan_ranked_by_f1, Gan_ranked_by_pr_auc, \
            Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names, Gan_ranked_by_vus_names = (
                gan_results[0], gan_results[1], gan_results[2], gan_results[3], gan_results[4]
            )
            logger.info("  ✓ [GAN] F1 names top-5: %s | Time=%.4fs", Gan_ranked_by_f1_names[:5],
                        timing_dict['3_GAN_Robustness'])
            logger.info("     [GAN] PR names top-5: %s", Gan_ranked_by_pr_auc_names[:5])
    else:
        reason = "skip_gan=True" if ("gan" in stages and skip_gan) else "not in --stages"
        logger.info(f"  ⏩ Sub-stage 6.3: GAN Robustness Testing SKIPPED ({reason})")
        # Empty placeholders already set above for skipped GAN
        timing_dict['3_GAN_Robustness'] = 0.0
        # --skip_gan, not a partial run that leaves GAN out of --stages.
        if explain and "gan" in stages:
            _retire_gan_explanation(dataset, entity)

    # --------------------------------------------
    # 4) Off-by-threshold (borderline sensitivity)
    # --------------------------------------------
    if "offby" in stages:
        logger.info("  📊 Sub-stage 6.4: Off-by-Threshold Testing...")
        mem_before = get_memory_usage_mb()
        start_time = time.time()
        # Use original un-injected data so synthetic spike labels don't cause single-class skips
        test_data_for_borderline = copy.deepcopy(test_data_gan if test_data_gan is not None else test_data)
        ranked_by_f1, ranked_by_pr_auc, ranked_by_f1_names_sensitivity, \
        ranked_by_pr_auc_names_sensitivity, ranked_by_vus_names_sensitivity = run_off_by_threshold(
            test_data_for_borderline, trained_models, still_usable(), dataset, entity,
            explain=explain, metrics=decision_metric
        )
        timing_dict['4_Borderline_Sensitivity'] = time.time() - start_time
        mem_after = get_memory_usage_mb()
        memory_dict['modules']['4_Borderline_Sensitivity'] = {
            'before': mem_before,
            'after': mem_after,
            'delta': mem_after - mem_before,
            'peak': _MEM_TRACE.peak_between(start_time, time.time()),
        }
        logger.info("  ✓ [Borderline] F1 names top-5: %s | Time=%.4fs", ranked_by_f1_names_sensitivity[:5],
                    timing_dict['4_Borderline_Sensitivity'])
        logger.info("     [Borderline] PR names top-5: %s", ranked_by_pr_auc_names_sensitivity[:5])

    # ---------------------------------
    # 5) Monte Carlo (noise stress test)
    # ---------------------------------
    if "montecarlo" in stages:
        logger.info("  📊 Sub-stage 6.5: Monte Carlo Simulation...")
        mem_before = get_memory_usage_mb()
        start_time = time.time()
        # Use original un-injected data for the same reason
        test_data_for_mc = copy.deepcopy(test_data_gan if test_data_gan is not None else test_data)
        monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR, \
        monte_carlo_ranked_models_VUS = run_monte_carlo_simulation(
            test_data_for_mc, trained_models, still_usable(), dataset, entity,
            n_simulations=2, noise_level=0.1, explain=explain, metrics=decision_metric,
        )
        timing_dict['5_Monte_Carlo'] = time.time() - start_time
        mem_after = get_memory_usage_mb()
        memory_dict['modules']['5_Monte_Carlo'] = {
            'before': mem_before,
            'after': mem_after,
            'delta': mem_after - mem_before,
            'peak': _MEM_TRACE.peak_between(start_time, time.time()),
        }
        logger.info("  ✓ [MonteCarlo] F1 names top-5: %s | Time=%.4fs", monte_carlo_ranked_models_F1[:5],
                    timing_dict['5_Monte_Carlo'])
        logger.info("     [MonteCarlo] PR names top-5: %s", monte_carlo_ranked_models_PR[:5])

    # Builder for the 11-item return tuple, shared by the partial-run early return
    # below and the full-run return at the end (captures current values at call time).
    def _result():
        return (
            (thompson_model_names[0] if thompson_model_names else None),
            robust_agg[1],
            full_aggregated[1],
            best_ensemble,
            individual_predictions,
            base_model_predictions_train,
            base_model_predictions_test,
            y_true_train,
            y_true_test,
            meta_model_type,
            {
                'ga': {'f1': best_f1, 'pr_auc': best_pr_auc, 'fitness': best_fitness,
                       'scores': ensemble_scores, 'y_true': y_true_test,
                       'meta_model': best_meta_model},
                'thompson': thompson_model_names,
                'gan': {
                    'f1_names': Gan_ranked_by_f1_names,
                    'pr_auc_names': Gan_ranked_by_pr_auc_names,
                    'vus_names': Gan_ranked_by_vus_names,
                    'f1_scores': Gan_ranked_by_f1,
                    'pr_auc_scores': Gan_ranked_by_pr_auc,
                    'best_model': Gan_ranked_by_f1_names[0] if len(Gan_ranked_by_f1_names) > 0 else 'N/A',
                    'best_f1': Gan_ranked_by_f1[0][1][0]['f1'] if len(Gan_ranked_by_f1) > 0 else 0.0,
                    'best_pr_auc': Gan_ranked_by_pr_auc[0][1][0]['pr_auc'] if len(Gan_ranked_by_pr_auc) > 0 else 0.0,
                },
                'borderline': {
                    'f1_names': ranked_by_f1_names_sensitivity,
                    'pr_auc_names': ranked_by_pr_auc_names_sensitivity,
                    'vus_names': ranked_by_vus_names_sensitivity,
                    'f1_scores': ranked_by_f1,
                    'pr_auc_scores': ranked_by_pr_auc,
                    'best_model': ranked_by_f1_names_sensitivity[0] if len(ranked_by_f1_names_sensitivity) > 0 else 'N/A',
                    'best_f1': ranked_by_f1[0][1][0]['f1'] if len(ranked_by_f1) > 0 else 0.0,
                    'best_pr_auc': ranked_by_pr_auc[0][1][0]['pr_auc'] if len(ranked_by_pr_auc) > 0 else 0.0,
                },
                'monte_carlo': {
                    'f1_names': monte_carlo_ranked_models_F1,
                    'pr_auc_names': monte_carlo_ranked_models_PR,
                    'vus_names': monte_carlo_ranked_models_VUS,
                    'best_model_f1': monte_carlo_ranked_models_F1[0] if len(monte_carlo_ranked_models_F1) > 0 else 'N/A',
                    'best_model_pr_auc': monte_carlo_ranked_models_PR[0] if len(monte_carlo_ranked_models_PR) > 0 else 'N/A'
                },
                'robust_agg': robust_agg,
                'full_aggregated': full_aggregated,
                'timing': timing_dict,
                'memory': memory_dict
            }
        )

    # Partial run (strict subset of stages): rank aggregation needs all robustness
    # tests + Thompson, so stop here and return what ran.
    if stages != ALL_STAGES:
        logger.info(f"  ⏩ Sub-stage 6.6: Rank Aggregation SKIPPED (partial run: {','.join(sorted(stages))})")
        memory_dict['final'] = get_memory_usage_mb()
        memory_dict['peak'] = get_peak_memory_mb()
        return _result()

    # -----------------------
    # 6) Rank Aggregations
    # -----------------------
    logger.info("  📊 Sub-stage 6.6: Rank Aggregation...")
    mem_before = get_memory_usage_mb()
    start_time = time.time()
    # Robust-only aggregation in the requested order: GAN → Borderline → Monte Carlo.
    # A fitness of exactly one of F1 / PR-AUC contributes only that ranking.
    _by_metric = {
        "f1": [("GAN_F1", Gan_ranked_by_f1_names),
               ("Borderline_F1", ranked_by_f1_names_sensitivity),
               ("MonteCarlo_F1", monte_carlo_ranked_models_F1)],
        "pr_auc": [("GAN_PR_AUC", Gan_ranked_by_pr_auc_names),
                   ("Borderline_PR_AUC", ranked_by_pr_auc_names_sensitivity),
                   ("MonteCarlo_PR_AUC", monte_carlo_ranked_models_PR)],
        "vus": [("GAN_VUS", Gan_ranked_by_vus_names),
                ("Borderline_VUS", ranked_by_vus_names_sensitivity),
                ("MonteCarlo_VUS", monte_carlo_ranked_models_VUS)],
    }
    _sources = []
    for _stage in range(3):
        for _m in ranking_metrics_for(decision_metric):
            _sources.append(_by_metric[_m][_stage])
    rank_source_names = [n for n, _ in _sources]
    test_for_rank = [r for _, r in _sources]
    # A stage that produced no ranking (skipped, failed soft, or too little
    # data) gets no vote: the aggregator drops empty rankings. The log names
    # those sources so the count printed here matches the sources that vote.
    _voting = [n for n, r in _sources if len(r) > 0]
    _silent = [n for n, r in _sources if len(r) == 0]
    logger.info(f"  → Consensus sources ({len(_voting)}): {', '.join(_voting)}"
                + (f" | no ranking from: {', '.join(_silent)}" if _silent else ""))
    robust_agg = enhanced_markov_chain_rank_aggregator_text(test_for_rank)
    explain_rank_aggregation(
        rankings=test_for_rank,
        source_names=rank_source_names,
        full_ranking=robust_agg[1],
        stage_name="robust",
        dataset=dataset, entity=entity, iteration=iteration, explain=explain,
    )

    # Final merge of robust aggregation vs Thompson Sampling
    full_ = [robust_agg[1], thompson_model_names]
    full_aggregated = enhanced_markov_chain_rank_aggregator_text(full_)
    explain_rank_aggregation(
        rankings=full_,
        source_names=["Robust_Aggregated", "Thompson_Sampling"],
        full_ranking=full_aggregated[1],
        stage_name="final",
        dataset=dataset, entity=entity, iteration=iteration, explain=explain,
    )
    timing_dict['6_Rank_Aggregation'] = time.time() - start_time
    mem_after = get_memory_usage_mb()
    memory_dict['modules']['6_Rank_Aggregation'] = {
        'before': mem_before,
        'after': mem_after,
        'delta': mem_after - mem_before,
        'peak': _MEM_TRACE.peak_between(start_time, time.time()),
    }
    logger.info("  ✓ [Aggregation] Time=%.4fs", timing_dict['6_Rank_Aggregation'])
    
    # Track final memory and peak
    memory_dict['final'] = get_memory_usage_mb()
    memory_dict['peak'] = get_peak_memory_mb()
    _pk = memory_dict['peak']
    logger.info(f"  💾 Final memory usage: {memory_dict['final']:.2f} MB "
                f"(Peak: {_pk:.2f} MB)" if _pk is not None
                else f"  💾 Final memory usage: {memory_dict['final']:.2f} MB "
                     f"(Peak: unavailable on this platform)")

    # -----------------------
    # Persist a concise report
    # -----------------------
    directory = results_dir("robust_aggregated", dataset, entity)
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(
        directory, f"robust_aggregated_results_{dataset}_{entity}_{iteration}.txt"
    )
    with open(output_file, 'w') as f:
        f.write("Summary of robust tests (order: GAN → Borderline → Monte Carlo):\n")
        f.write("\n[GAN]\n")
        f.write(f"{Gan_ranked_by_f1_names}\n{Gan_ranked_by_pr_auc_names}\n")
        f.write("\n[Borderline / Off-by-threshold]\n")
        f.write(f"{ranked_by_f1_names_sensitivity}\n{ranked_by_pr_auc_names_sensitivity}\n")
        f.write("\n[Monte Carlo]\n")
        f.write(f"{monte_carlo_ranked_models_F1}\n{monte_carlo_ranked_models_PR}\n")
        f.write("\n[Robust rank aggregate]\n")
        f.write(f"{robust_agg}\n")
        f.write("\n[Final aggregate vs Thompson]\n")
        f.write(f"{full_aggregated}\n")

    # Return extended tuple with all necessary data for comprehensive results
    return _result()


def run_model_selection_algorithms_2(train_data, test_data, dataset, entity, iteration, trained_models, model_list=None, test_data_gan=None, skip_gan=False, explain=False, decision_metric=DEFAULT_DECISION_METRICS):
    """
    PARALLEL VERSION: Runs model selection algorithms concurrently using ThreadPoolExecutor.
    
    Executes algorithms in parallel:
      1) GA (genetic algorithm for stacking ensemble)
      2) Thompson Sampling (online LinTS)
      3) GAN robustness test (optional, can be skipped)
      4) Off-by-threshold (borderline sensitivity)
      5) Monte Carlo (noise stress test)
    
    Parameters
    ----------
    trained_models : dict
        Dictionary of trained model instances
    model_list : list[str], optional
        List of model names to use. If None, uses global algorithm_list_instances.
    test_data_gan : Dataset, optional
        Separate dataset for GAN test (should be full original data without injection).
        GAN creates its own perturbations internally. If None, uses test_data.
    skip_gan : bool, optional
        If True, skips GAN test for faster re-optimization in online phase.
        
    Returns
    -------
    Same 11-item tuple as run_model_selection_algorithms_1
    """
    # Use provided model list or fall back to global
    models_to_use = model_list if model_list is not None else algorithm_list_instances
    
    logger.info("  🚀 Starting PARALLEL model selection (5 algorithms concurrently)...")
    logger.info("  ✨ VERSION CHECK: Using run_model_selection_algorithms_2 with deepcopy fix")
    timing_dict = {}
    memory_dict = {'modules': {}}
    overall_start = time.time()
    
    # Track initial memory. The sampler runs alongside from here, so every
    # module gets a PEAK over its own window rather than two endpoint samples.
    _MEM_TRACE.start()
    initial_memory = get_memory_usage_mb()
    memory_dict['initial'] = initial_memory
    memory_dict['llm_server_initial'] = llm_server_memory_mb()
    logger.info(f"  💾 Initial memory usage: {initial_memory:.2f} MB")
    
    # CRITICAL: Create independent copies of test_data for each algorithm
    # This prevents race conditions when algorithms modify data during processing
    logger.info("     📋 Creating independent data copies...")
    test_data_ga = copy.deepcopy(test_data)
    test_data_thompson = copy.deepcopy(test_data)
    # GAN uses separate dataset (full original) to avoid circularity
    test_data_gan_copy = copy.deepcopy(test_data_gan if test_data_gan is not None else test_data) if not skip_gan else None
    test_data_borderline = copy.deepcopy(test_data_gan if test_data_gan is not None else test_data)
    test_data_montecarlo = copy.deepcopy(test_data_gan if test_data_gan is not None else test_data)
    num_tasks = 4 if skip_gan else 5
    logger.info(f"     ✓ Created {num_tasks} independent data copies for thread safety")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as executor:
        task_names = "GA | Thompson | Borderline | Monte Carlo" if skip_gan else "GA | Thompson | GAN | Borderline | Monte Carlo"
        logger.info(f"     Launching: {task_names}")
        
        # Submit all algorithms with their own data copies
        ga_future = executor.submit(
            genetic_algorithm,
            dataset, entity, train_data, test_data_ga,
            models_to_use, trained_models,
            population_size=20, generations=20, meta_model_type='rf', mutation_rate=0.1,
            explain=explain,
            metric=decision_metric,
            vus_win=vus_window(test_data_ga.entities[0].Y),
        )
        thompson_future = executor.submit(
            run_linear_thompson_sampling,
            test_data=test_data_thompson,
            trained_models=trained_models,
            model_names=models_to_use,
            dataset=dataset,
            entity=entity,
            iterations=50,
            iteration=iteration,
            explain=explain,
        )
        
        if not skip_gan:
            # Through the fail-soft wrapper: an exception in this worker would
            # otherwise re-raise from gan_future.result() below and lose the
            # other four stages' finished work along with the GAN.
            gan_future = executor.submit(
                _run_gan_or_skip,
                test_data_gan_copy, trained_models, models_to_use,
                dataset, entity, explain=explain, metrics=decision_metric
            )
        
        borderline_future = executor.submit(
            run_off_by_threshold,
            test_data_borderline, trained_models, models_to_use,
            dataset, entity, explain=explain, metrics=decision_metric
        )
        monte_carlo_future = executor.submit(
            run_monte_carlo_simulation,
            test_data_montecarlo, trained_models, models_to_use,
            dataset, entity, 2, 0.1, explain=explain, metrics=decision_metric
        )

        # Collect results
        logger.info("     ⏳ Waiting for parallel tasks...")
        
        best_ensemble, best_f1, best_pr_auc, best_fitness, \
        individual_predictions, base_model_predictions_train, base_model_predictions_test, \
        y_true_train, y_true_test, meta_model_type, ensemble_scores = ga_future.result()
        timing_dict['1_GA'] = time.time() - overall_start
        mem_ga = get_memory_usage_mb()
        memory_dict['modules']['1_GA'] = {
            'before': initial_memory,
            'after': mem_ga,
            'delta': mem_ga - initial_memory,
            # Concurrent stages share one process, so a per-stage peak is not
            # separable here; this is the peak over the whole parallel region.
            'peak': _MEM_TRACE.peak_between(overall_start, time.time()),
            'peak_shared': True,
        }
        logger.info("     ✓ GA: ensemble=%s | F1=%.4f | PR-AUC=%.4f", best_ensemble, best_f1, best_pr_auc)
        
        thompson_model_names = thompson_future.result()
        timing_dict['2_Thompson'] = time.time() - overall_start
        mem_thompson = get_memory_usage_mb()
        memory_dict['modules']['2_Thompson'] = {
            'before': initial_memory,
            'after': mem_thompson,
            'delta': mem_thompson - initial_memory,
            'peak': _MEM_TRACE.peak_between(overall_start, time.time()),
            'peak_shared': True,
        }
        logger.info("     ✓ Thompson: top-5=%s", thompson_model_names[:5])
        
        # None when the stage failed soft; it then takes the --skip_gan branch.
        gan_results = None if skip_gan else gan_future.result()
        if gan_results is not None:
            Gan_ranked_by_f1, Gan_ranked_by_pr_auc, Gan_ranked_by_f1_names, \
            Gan_ranked_by_pr_auc_names, Gan_ranked_by_vus_names = gan_results
            timing_dict['3_GAN'] = time.time() - overall_start
            mem_gan = get_memory_usage_mb()
            memory_dict['modules']['3_GAN'] = {
                'before': initial_memory,
                'after': mem_gan,
                'delta': mem_gan - initial_memory,
                'peak': _MEM_TRACE.peak_between(overall_start, time.time()),
                'peak_shared': True,
            }
            logger.info("     ✓ GAN: F1 top-5=%s", Gan_ranked_by_f1_names[:5])
        else:
            logger.info("     ⏩ GAN: SKIPPED (%s)", "skip_gan=True" if skip_gan
                        else "the stage failed; see the warning above")
            Gan_ranked_by_f1 = []
            Gan_ranked_by_pr_auc = []
            Gan_ranked_by_f1_names = []
            Gan_ranked_by_pr_auc_names = []
            Gan_ranked_by_vus_names = []
            # A failed stage still took time to fail; only --skip_gan costs nothing.
            timing_dict['3_GAN'] = 0.0 if skip_gan else time.time() - overall_start
            # This pipeline always runs every stage, so both causes count.
            if explain:
                _retire_gan_explanation(dataset, entity)

        # run_off_by_threshold returns five values. This used to unpack seven,
        # because a merge had left a duplicate of the first two names in front.
        # Every full --parallel run then raised "not enough values to unpack
        # (expected 7, got 5)" here, before rank aggregation.
        ranked_by_f1, ranked_by_pr_auc, ranked_by_f1_names_sensitivity, \
        ranked_by_pr_auc_names_sensitivity, ranked_by_vus_names_sensitivity = borderline_future.result()
        timing_dict['4_Borderline'] = time.time() - overall_start
        mem_borderline = get_memory_usage_mb()
        memory_dict['modules']['4_Borderline'] = {
            'before': initial_memory,
            'after': mem_borderline,
            'delta': mem_borderline - initial_memory,
            'peak': _MEM_TRACE.peak_between(overall_start, time.time()),
            'peak_shared': True,
        }
        logger.info("     ✓ Borderline: F1 top-5=%s", ranked_by_f1_names_sensitivity[:5])
        
        monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR, \
        monte_carlo_ranked_models_VUS = monte_carlo_future.result()
        timing_dict['5_MonteCarlo'] = time.time() - overall_start
        mem_mc = get_memory_usage_mb()
        memory_dict['modules']['5_MonteCarlo'] = {
            'before': initial_memory,
            'after': mem_mc,
            'delta': mem_mc - initial_memory,
            'peak': _MEM_TRACE.peak_between(overall_start, time.time()),
            'peak_shared': True,
        }
        logger.info("     ✓ MonteCarlo: F1 top-5=%s", monte_carlo_ranked_models_F1[:5])

    timing_dict['0_Parallel_Total'] = time.time() - overall_start
    logger.info("  ✅ All 5 algorithms completed in %.2fs (parallel)", timing_dict['0_Parallel_Total'])

    # Rank Aggregation
    logger.info("  📊 Sub-stage 6.6: Rank Aggregation...")
    mem_before_agg = get_memory_usage_mb()
    agg_start = time.time()
    test_for_rank = [
        Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names,
        ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity,
        monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR,
    ]
    robust_agg = enhanced_markov_chain_rank_aggregator_text(test_for_rank)
    explain_rank_aggregation(
        rankings=test_for_rank,
        source_names=["GAN_F1", "GAN_PR_AUC",
                      "Borderline_F1", "Borderline_PR_AUC",
                      "MonteCarlo_F1", "MonteCarlo_PR_AUC"],
        full_ranking=robust_agg[1],
        stage_name="robust",
        dataset=dataset, entity=entity, iteration=iteration, explain=explain,
    )
    full_ = [robust_agg[1], thompson_model_names]
    full_aggregated = enhanced_markov_chain_rank_aggregator_text(full_)
    explain_rank_aggregation(
        rankings=full_,
        source_names=["Robust_Aggregated", "Thompson_Sampling"],
        full_ranking=full_aggregated[1],
        stage_name="final",
        dataset=dataset, entity=entity, iteration=iteration, explain=explain,
    )
    timing_dict['6_Aggregation'] = time.time() - agg_start
    mem_after_agg = get_memory_usage_mb()
    memory_dict['modules']['6_Aggregation'] = {
        'before': mem_before_agg,
        'after': mem_after_agg,
        'delta': mem_after_agg - mem_before_agg,
        'peak': _MEM_TRACE.peak_between(agg_start, time.time()),
    }
    logger.info("  ✓ Aggregation: %.2fs", timing_dict['6_Aggregation'])
    
    # Track final memory and peak
    memory_dict['final'] = get_memory_usage_mb()
    memory_dict['peak'] = get_peak_memory_mb()
    _pk = memory_dict['peak']
    logger.info(f"  💾 Final memory usage: {memory_dict['final']:.2f} MB "
                f"(Peak: {_pk:.2f} MB)" if _pk is not None
                else f"  💾 Final memory usage: {memory_dict['final']:.2f} MB "
                     f"(Peak: unavailable on this platform)")

    # Persist results
    directory = results_dir("robust_aggregated", dataset, entity)
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(
        directory, f"robust_aggregated_results_{dataset}_{entity}_{iteration}.txt"
    )
    with open(output_file, 'w') as f:
        f.write("Summary of robust tests (PARALLEL execution):\n")
        f.write("\n[GAN]\n")
        f.write(f"{Gan_ranked_by_f1_names}\n{Gan_ranked_by_pr_auc_names}\n")
        f.write("\n[Borderline]\n")
        f.write(f"{ranked_by_f1_names_sensitivity}\n{ranked_by_pr_auc_names_sensitivity}\n")
        f.write("\n[Monte Carlo]\n")
        f.write(f"{monte_carlo_ranked_models_F1}\n{monte_carlo_ranked_models_PR}\n")
        f.write("\n[Robust rank aggregate]\n")
        f.write(f"{robust_agg}\n")
        f.write("\n[Final aggregate vs Thompson]\n")
        f.write(f"{full_aggregated}\n")

    # Return 11-item tuple matching sequential version
    return (
        thompson_model_names[0],
        robust_agg[1],
        full_aggregated[1],
        best_ensemble,
        individual_predictions,
        base_model_predictions_train,
        base_model_predictions_test,
        y_true_train,
        y_true_test,
        meta_model_type,
        {
            'ga': {'f1': best_f1, 'pr_auc': best_pr_auc, 'fitness': best_fitness,
                   'scores': ensemble_scores, 'y_true': y_true_test},
            'thompson': thompson_model_names,
            'gan': {
                'f1_names': Gan_ranked_by_f1_names, 
                'pr_auc_names': Gan_ranked_by_pr_auc_names,
                'vus_names': Gan_ranked_by_vus_names,
                'f1_scores': Gan_ranked_by_f1,
                'pr_auc_scores': Gan_ranked_by_pr_auc,
                'best_model': Gan_ranked_by_f1_names[0] if len(Gan_ranked_by_f1_names) > 0 else 'N/A',
                'best_f1': Gan_ranked_by_f1[0][1][0]['f1'] if len(Gan_ranked_by_f1) > 0 else 0.0,
                'best_pr_auc': Gan_ranked_by_pr_auc[0][1][0]['pr_auc'] if len(Gan_ranked_by_pr_auc) > 0 else 0.0,
            },
            'borderline': {
                'f1_names': ranked_by_f1_names_sensitivity, 
                'pr_auc_names': ranked_by_pr_auc_names_sensitivity,
                'vus_names': ranked_by_vus_names_sensitivity,
                'f1_scores': ranked_by_f1,
                'pr_auc_scores': ranked_by_pr_auc,
                'best_model': ranked_by_f1_names_sensitivity[0] if len(ranked_by_f1_names_sensitivity) > 0 else 'N/A',
                'best_f1': ranked_by_f1[0][1][0]['f1'] if len(ranked_by_f1) > 0 else 0.0,
                'best_pr_auc': ranked_by_pr_auc[0][1][0]['pr_auc'] if len(ranked_by_pr_auc) > 0 else 0.0,
            },
            'monte_carlo': {
                'f1_names': monte_carlo_ranked_models_F1, 
                'pr_auc_names': monte_carlo_ranked_models_PR,
                'vus_names': monte_carlo_ranked_models_VUS,
                'best_model_f1': monte_carlo_ranked_models_F1[0] if len(monte_carlo_ranked_models_F1) > 0 else 'N/A',
                'best_model_pr_auc': monte_carlo_ranked_models_PR[0] if len(monte_carlo_ranked_models_PR) > 0 else 'N/A'
            },
            'robust_agg': robust_agg,
            'full_aggregated': full_aggregated,
            'timing': timing_dict,
            'memory': memory_dict
        }
    )

# ------------------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------------------

def save_current_selection(dataset, entity, window_idx, best_ensemble, best_single_model, 
                          meta_model_type, trained_models, individual_predictions,
                          base_model_predictions_train, base_model_predictions_test,
                          y_true_train, y_true_test):
    """
    Save the current model selection state to CSV for tracking online adaptations.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    entity : str
        Entity ID
    window_idx : int
        Current window index (0 for offline phase, N for online window N)
    best_ensemble : list
        List of model names in the best ensemble
    best_single_model : str
        Name of the best single model
    meta_model_type : str
        Type of meta-model used ('rf', 'lr', etc.)
    trained_models : dict
        Dictionary of trained model instances
    individual_predictions : dict
        Individual model predictions
    base_model_predictions_train/test : array
        Base model predictions on train/test sets
    y_true_train/test : array
        True labels for train/test sets
    """
    import csv
    import pickle
    from pathlib import Path
    
    # Create output directory
    output_dir = Path(results_dir("online", dataset, entity))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV file for tracking selections over time
    csv_file = output_dir / f"online_selections_window_{window_idx}.csv"
    
    # Save metadata to CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Field', 'Value'])
        writer.writerow(['Window_Index', window_idx])
        writer.writerow(['Best_Single_Model', best_single_model])
        writer.writerow(['Best_Ensemble_Size', len(best_ensemble)])
        writer.writerow(['Best_Ensemble_Models', ';'.join(best_ensemble)])
        writer.writerow(['Meta_Model_Type', meta_model_type])
        writer.writerow(['Timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    
    logger.info(f"  💾 Saved current selection state to {csv_file}")
    
    # Optionally save complete state as pickle for later restoration
    # (Commented out by default to save space - uncomment if needed)
    # state_file = output_dir / f"selection_state_window_{window_idx}.pkl"
    # state = {
    #     'window_idx': window_idx,
    #     'best_ensemble': best_ensemble,
    #     'best_single_model': best_single_model,
    #     'meta_model_type': meta_model_type,
    #     'individual_predictions': individual_predictions,
    #     'base_model_predictions_train': base_model_predictions_train,
    #     'base_model_predictions_test': base_model_predictions_test,
    #     'y_true_train': y_true_train,
    #     'y_true_test': y_true_test,
    # }
    # with open(state_file, 'wb') as f:
    #     pickle.dump(state, f)
    # logger.info(f"  💾 Saved complete state to {state_file}")

# The per-window misclassification counts that `find_num_falses` wrote to
# robust_aggregated/new_robust_aggregated_results_*.txt now travel in each
# window's record of results/online/<dataset>/<entity>/<scenario>.json
# (`misclassified`). One file per window index could not hold four scenarios,
# and its ensemble count cast meta-model PROBABILITIES to int, so it counted
# the positives rather than the errors.

# ------------------------------------------------------------------------------
# Re-optimization (runs in its own process; see Model_Selection/online_reopt.py)
# ------------------------------------------------------------------------------

# What a re-optimisation process calls (Model_Selection/online_reopt.py).
REOPT_TARGET = "app:perform_reoptimization_task"


def perform_reoptimization_task(*, window_idx, scenario, train_data, sliding,
                                sliding_raw, dataset, entity, models_dir,
                                model_names, decision_metric, seed,
                                label_source="injected"):
    """
    One re-optimisation of the online phase, run inside its own child process
    (Model_Selection/online_reopt.py), which seeds every generator with `seed`
    and redirects all results into a scratch directory before calling this.

    It re-runs the offline model-selection pipeline — GAN skipped, explanations
    off, the sequential path — on what the offline phase selected on, slid
    forward in time (online_scoring.sliding_series): `sliding` is the offline
    phase's injected series followed by the injected online samples seen up
    to the triggering window, each sample once, cut to the offline length;
    `sliding_raw` is the same window before injection, which the robustness
    stages score, as they do offline. The labels are the injected ones the
    online windows are scored on.

    With `label_source` "real" nothing is injected: `sliding` and
    `sliding_raw` are the same raw series with its TRUE labels (the offline
    part slid forward plus the online samples seen so far, each once), so
    every stage selects on what was really labelled — a supervised
    re-selection that assumes the labels are fed back. The caller has checked
    that it holds a labelled anomaly (online_scoring.require_labelled_anomaly).

    The detectors are loaded from the checkpoints the pipeline loaded
    (`model_names` from `models_dir`); the pipeline already probed them, so
    they are not probed again. Returns the new Selection, its ensemble carrying
    the GA's own fitted meta-model.
    """
    global trained_models
    from Model_Selection.online_scoring import fit_selection

    started = time.time()
    real = label_source == ONLINE_LABELS_REAL
    logger.info(f"  🔄 [REOPT] Re-optimization for scenario {scenario}, window "
                f"{window_idx}: {sliding.entities[0].Y.shape[1]} "
                f"{'real-labelled' if real else 'injected'} samples "
                f"({int(np.sum(np.asarray(sliding.entities[0].labels) > 0))} anomalous), "
                f"{len(model_names)} detectors, seed {seed}")
    trained_models, excluded = load_trained_models(model_names, models_dir)
    missing = [m for m in model_names if m not in trained_models]
    if missing:
        raise RuntimeError(f"could not load {', '.join(missing)} from {models_dir}"
                           + (f" ({'; '.join(f'{k}: {v}' for k, v in excluded.items())})"
                              if excluded else ""))
    try:
        (_, _, full_aggregated, best_ensemble, individual_predictions,
         base_model_predictions_train, _, y_true_train, _, meta_model_type,
         extra) = run_model_selection_algorithms_1(
            train_data, sliding, dataset, entity, iteration=window_idx,
            model_list=list(model_names), test_data_gan=sliding_raw,
            skip_gan=True, explain=False, decision_metric=decision_metric,
            real_labels=real)
    finally:
        # The GA's plotting helpers open figures they never close.
        plt.close('all')

    if not best_ensemble:
        raise ValueError("the re-optimization selected no ensemble")
    single = (full_aggregated[0] if isinstance(full_aggregated, (list, tuple))
              else full_aggregated)
    if not single:
        raise ValueError("the re-optimization ranked no single model")
    selection = fit_selection(
        best_ensemble, single, meta_model_type, individual_predictions,
        base_model_predictions_train, y_true_train, origin="reoptimization",
        origin_window=window_idx, train_data=train_data, trained_models=trained_models,
        meta_model=(extra.get('ga') or {}).get('meta_model'), seed=seed)
    logger.info(f"  ✓ [REOPT] Re-optimization for scenario {scenario}, window "
                f"{window_idx} completed in {time.time() - started:.2f}s: "
                f"ensemble={selection.ensemble}, single={selection.single}")
    return selection


def run_online_phase(*, args, dataset, entity, train_data, trained_models,
                     loaded_model_names, models_dir, offline_copies, offline_raw,
                     stream_raw, feasibility, window_size, step_size, total_samples,
                     offline_size, online_size, anomaly_list, anomaly_rate,
                     best_ensemble, best_single_model, meta_model_type, ga_meta_model,
                     individual_predictions, base_model_predictions_train,
                     y_true_train, offline_summary, vus_win, iteration, offline_time):
    """
    The online phase: the online windows replayed under each requested scenario.

    The synthetic anomalies are injected once into the whole online stream and
    every window is cut from it — or, with real labels (args
    'online_label_source' == "real"), nothing is injected and every window is
    the raw test slice with its true labels — and every scenario starts from its own copy
    of the offline selection, so scenarios differ in their re-optimisation
    policy and nothing else. A re-optimisation runs in a child process and is
    switched in exactly --reopt_delay windows after its trigger. Results go to
    results/online/<dataset>/<entity>/ (schema: Model_Selection/online_phase.py).

    `offline_copies` are the offline phase's injected copies (Inject with
    return_copies); `offline_raw` and `stream_raw` are (Y, labels, mask) of the
    offline part and of the online stream before the synthetic anomalies.

    Never raises. The offline results are already written by the time this
    runs, so a failure here is logged as `ONLINE PHASE FAILED`, recorded in
    summary.json, and the run's offline outcome stands.
    """
    from Model_Selection.online_phase import (OnlineResultsWriter, run_online,
                                              scenario_params)
    from Model_Selection.online_policy import PolicyParams
    from Model_Selection.online_reopt import CompletedReopt, ProcessReopt
    from Model_Selection.online_scoring import (WindowScorer, build_online_stream,
                                                build_online_windows, fit_selection,
                                                identity_copy, isolated_models,
                                                random_selection, real_label_stream,
                                                require_labelled_anomaly,
                                                sliding_series)
    from Utils.pipeline_spec import count_windows, format_decision_metrics

    scenarios = tuple(args['online_scenarios'])
    params = PolicyParams(update_interval=args['update_interval'],
                          drop_pct=args['drop_pct'],
                          adaptive_window=args['adaptive_window'],
                          adaptive_k=args['adaptive_k'],
                          reopt_delay=args['reopt_delay'])
    reopt_timeout = args.get('reopt_timeout')
    # The fitness the offline phase OPTIMISED, not the possibly narrowed one
    # its final decision fell back to.
    decision_metric = args.get('decision_metrics', DEFAULT_DECISION_METRICS)
    seed = args.get('random_seed')
    seed = 13 if seed is None else int(seed)
    stream_Y, stream_labels, stream_mask = stream_raw
    stream_len = int(stream_Y.shape[1])
    available = count_windows(stream_len, window_size, step_size) if feasibility['ok'] else 0
    max_windows = args.get('max_online_windows')
    n_windows = available if max_windows is None else min(available, max_windows)
    indices = list(range(n_windows))           # window 0 is the first online window
    regime = bool(args.get('inject_online_regime'))
    label_source = args.get('online_label_source') or "injected"
    real = label_source == ONLINE_LABELS_REAL

    # Resolved here: every write of this phase goes to this one directory.
    online_dir = results_dir("online", dataset, entity)
    header = {
        "dataset": dataset,
        "dataset_label": dataset_label(dataset),
        "entity": entity,
        "strategy": args.get('strategy'),
        "decision_metric": {"spec": format_decision_metrics(decision_metric),
                            "label": decision_metric_label(decision_metric),
                            "formula": decision_metric_formula(decision_metric)},
        "window_size": int(window_size),
        "step_size": int(step_size),
        "n_windows": n_windows,
        "n_windows_available": available,
        "first_window_index": 0,
        "total_samples": int(total_samples),
        "offline_size": int(offline_size),
        "online_size": int(online_size),
        "online_samples": stream_len,
        "window_coordinates": "online_stream" if regime else "test_series",
        "inject_online_regime": regime,
        # What the windows are scored on, and whether anything was injected
        # into the online stream: "real" = the test series' own labels and
        # no online injection (`injection` stays null); "injected" = the
        # synthetic anomalies below.
        "label_source": label_source,
        "label_source_label": ONLINE_LABEL_SOURCE_LABELS.get(label_source, label_source),
        "n_windows_scorable": None,
        "n_windows_with_anomaly": None,
        # The offline phase's injection, which the online stream repeats
        # only when label_source is "injected".
        "anomaly_type": anomaly_list[0] if len(anomaly_list) == 1 else list(anomaly_list),
        "anomaly_rate": anomaly_rate,
        "injection": None,
        "seed": seed,
        "vus_window": vus_win if "vus" in metrics_required(decision_metric) else None,
        "reopt_delay": params.reopt_delay,
        "reopt_timeout": reopt_timeout,
        "feasibility": feasibility,
        "offline_selection": offline_summary,
    }
    writer = None
    try:
        writer = OnlineResultsWriter(online_dir, header, scenarios,
                                     scenario_params=scenario_params(scenarios, params),
                                     n_windows_planned=n_windows)
        if not feasibility['ok'] or n_windows < 1:
            reason = feasibility.get('reason') or (
                f"{available} window(s) of {window_size} samples fit in the "
                f"{stream_len}-sample online stream.")
            writer.finish("skipped", reason)
            logger.warning(f"⚠️  ONLINE PHASE SKIPPED: reason={reason}")
            return

        logger.info(f"  📊 Online phase: {n_windows} of {available} windows, "
                    f"scenarios={', '.join(scenarios)}, M={params.update_interval}, "
                    f"X={params.drop_pct:g}%, N={params.adaptive_window}, "
                    f"k={params.adaptive_k:g}, L={params.reopt_delay}, "
                    f"reopt_timeout={reopt_timeout if reopt_timeout else 'none'}")
        if args.get('parallel'):
            logger.info("  ℹ️  Re-optimizations run the sequential pipeline in their "
                        "own process, whatever --parallel says.")

        if real:
            # Nothing is injected online: the stream is the online part with
            # its true labels, and a re-optimisation slides the RAW offline
            # part forward, so every label it selects on is a real one.
            stream = real_label_stream(stream_Y, stream_labels, stream_mask)
            reopt_copies = [identity_copy(offline_raw[0], offline_raw[1])]
            writer.summary["injection"] = None
        else:
            stream = build_online_stream(stream_Y, stream_labels, stream_mask,
                                         anomaly_list, anomaly_rate, seed)
            reopt_copies = offline_copies
            writer.summary["injection"] = {"seed": seed, "n_copies": len(stream.copies),
                                           "copies": stream.copy_params()}
        windows = build_online_windows(stream, indices, offset=offline_size,
                                       step_size=step_size, window_size=window_size,
                                       name=f"{entity}_online")
        # The live monitor's stream files (Model_Selection/online_stream.py):
        # the series as replayed, its labels, and per window what each
        # scenario's detectors said. Optional: the phase runs without them.
        stream_writer = None
        try:
            from Model_Selection.online_stream import StreamWriter
            show_real = bool(args.get('labels_available')) and not regime
            stream_writer = StreamWriter(
                online_dir, dataset=dataset, entity=entity, label_source=label_source,
                offline_Y=offline_raw[0],
                offline_real_labels=offline_raw[1] if show_real else None,
                online_copies=stream.copies,
                online_real_labels=stream.raw_labels if show_real else None,
                offline_size=offline_size, window_size=window_size, step_size=step_size,
                n_windows=n_windows,
                window_coordinates="online_stream" if regime else "test_series",
                injected=not real)
        except Exception as e:
            logger.warning(f"⚠ Live-monitor stream files not written: {type(e).__name__}: {e}")
        writer.summary["n_windows_scorable"] = sum(1 for w in windows if w.scorable)
        writer.summary["n_windows_with_anomaly"] = sum(1 for w in windows
                                                       if (w.n_anomalous or 0) > 0)
        writer.write_summary()
        logger.info(f"  🏷️  Online labels: {label_source} — "
                    f"{writer.summary['n_windows_scorable']} of {len(windows)} windows "
                    f"scorable (both classes), {writer.summary['n_windows_with_anomaly']} "
                    f"hold a{' labelled' if real else 'n injected'} anomaly")
        # The stream scores its own copies: scoring has side effects on a model
        # object, and the pipeline's `trained_models` stay as the offline
        # phase left them.
        stream_models = isolated_models(trained_models)
        offline_selection = fit_selection(
            best_ensemble, best_single_model, meta_model_type, individual_predictions,
            base_model_predictions_train, y_true_train, origin="offline",
            train_data=train_data, trained_models=stream_models,
            meta_model=ga_meta_model, seed=seed)
        writer.summary["offline_selection"] = {
            **(offline_summary or {}),
            "ensemble": {**((offline_summary or {}).get("ensemble") or {}),
                         "members": list(offline_selection.ensemble),
                         "meta_model_source": offline_selection.meta_model_source}}
        writer.write_summary()
        baseline_selection = None
        if "fixed_random" in scenarios:
            pool = list(individual_predictions or loaded_model_names)
            baseline_selection = random_selection(
                pool, seed, meta_model_type, individual_predictions,
                base_model_predictions_train, y_true_train,
                train_data=train_data, trained_models=stream_models)
            logger.info(f"  🎲 Fixed random baseline: single={baseline_selection.single}, "
                        f"ensemble={baseline_selection.ensemble}")

        def initial_selection(key):
            # A deep copy per scenario: nothing one scenario does to its
            # selection can reach the next one's starting point.
            start = baseline_selection if key == "fixed_random" else offline_selection
            return copy.deepcopy(start)

        # A re-optimisation's input depends only on its triggering window (and
        # the detectors still usable), and its seed is seed + window, so a
        # second scenario triggering at the same window reuses the first one's
        # result instead of computing it again.
        reopt_cache = {}

        def start_reopt(window, selection, scenario):
            models = [m for m in loaded_model_names if m not in timed_out_detectors()]
            key = (window.index, tuple(models))
            if key in reopt_cache:
                value, duration, first = reopt_cache[key]
                return CompletedReopt(value, duration, reused_from=first)
            injected, raw = sliding_series(reopt_copies, offline_raw, stream,
                                           window.online_end, name=f"{entity}_sliding")
            if real:
                # Raises before anything is forked: recorded as a re-optimisation
                # that could not start, and the next trigger may try again.
                require_labelled_anomaly(
                    injected, what=f"the {injected.entities[0].Y.shape[1]}-sample "
                                   f"series seen up to window {window.index}")

            def remember(value, duration):
                reopt_cache.setdefault(key, (value, duration, scenario))

            return ProcessReopt(
                REOPT_TARGET,
                dict(window_idx=window.index, scenario=scenario, train_data=train_data,
                     sliding=injected, sliding_raw=raw, dataset=dataset, entity=entity,
                     models_dir=models_dir, model_names=models,
                     decision_metric=decision_metric, seed=seed + window.index,
                     label_source=label_source),
                workdir=os.path.join(online_dir, "_reopt_scratch", f"{scenario}_w{window.index}"),
                seed=seed + window.index, timeout=reopt_timeout, on_success=remember)

        online_start = time.time()
        # ONE scorer for every scenario: it scores each (window, detector) pair
        # once, so all scenarios read the same numbers for it.
        scorer = WindowScorer(stream_models, decision_metric, seed=seed, vus_win=vus_win)
        try:
            results = run_online(scenarios, params, windows, initial_selection,
                                 scorer, start_reopt, writer, log=logger,
                                 on_window=(stream_writer.record if stream_writer else None))
            logger.info(f"  🧮 Detector scorings on the stream: {scorer.computed} run, "
                        f"{scorer.reused} reused from an earlier scenario or member.")
        finally:
            # Each re-optimisation removes its own scratch directory; this
            # removes their parent once it is empty.
            try:
                os.rmdir(os.path.join(online_dir, "_reopt_scratch"))
            except OSError:
                pass
        online_total = time.time() - online_start
    except Exception as e:
        if writer is not None:
            try:
                writer.finish("failed", f"{type(e).__name__}: {e}")
            except Exception:
                pass
        logger.error(f"❌ ONLINE PHASE FAILED: error={type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        return

    logger.info(f"⏱️  Online Phase Total Time: {online_total:.2f}s ({online_total/60:.2f} min)")
    try:
        timing_file = _write_online_timing(
            dataset, entity, iteration, available, n_windows, scenarios, params,
            results, online_total, offline_time)
        logger.info(f"📝 Online results: {online_dir} | timing: {timing_file}")
    except OSError as e:
        logger.error(f"Could not write the online timing summary: {e}")


def _write_online_timing(dataset, entity, iteration, available, n_windows, scenarios,
                         params, results, online_total, offline_time):
    """The online phase's timing summary, beside the comprehensive report.

    Counts the windows each scenario actually processed; this file used to
    report every available window even when --max_online_windows cut the
    stream short.
    """
    timing_file = os.path.join(results_dir("comprehensive", dataset, entity),
                               "online_phase_timing.txt")
    os.makedirs(os.path.dirname(timing_file), exist_ok=True)
    with open(timing_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RAMSeS Online Phase Timing Summary\n")
        f.write("="*80 + "\n")
        f.write(f"Dataset: {dataset_label(dataset)}\n")
        f.write(f"Entity: {entity}\n")
        f.write(f"Iteration: {iteration}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Windows Available: {available}\n")
        f.write(f"Windows Processed per Scenario: {n_windows}\n")
        f.write(f"Scenarios: {', '.join(scenarios)}\n")
        f.write(f"Update Interval (M): {params.update_interval}\n")
        f.write(f"Drop (X): {params.drop_pct:g}%\n")
        f.write(f"Adaptive Window (N): {params.adaptive_window}\n")
        f.write(f"Adaptive k: {params.adaptive_k:g}\n")
        f.write(f"Re-optimization Delay (L): {params.reopt_delay}\n\n")
        for row in results:
            f.write(f"[{row['key']}] windows={row['n_windows']} "
                    f"triggers={row['n_triggers']} started={row['n_reopts_started']} "
                    f"applied={row['n_reopts_applied']} suppressed={row['n_suppressed']} "
                    f"failed={row['n_reopts_failed']} "
                    f"not_applied={row['n_reopts_not_applied']} "
                    f"reused={row['n_reopts_reused']} "
                    f"reopt={row['total_reopt_seconds']:.2f}s "
                    f"waited={row['total_wait_s']:.2f}s "
                    f"stream={row['stream_time_s'] or 0:.2f}s "
                    f"wall={row['wall_time_s'] or 0:.2f}s "
                    f"per-window={(row['stream_time_s'] or 0)/max(row['n_windows'], 1):.2f}s\n")
        f.write(f"\nTotal Online Time: {online_total:.2f}s ({online_total/60:.2f} min)\n")
        f.write(f"Offline Phase Time: {offline_time:.2f}s ({offline_time/60:.2f} min)\n")
        f.write(f"Grand Total Time: {online_total + offline_time:.2f}s "
                f"({(online_total + offline_time)/60:.2f} min)\n")
        f.write("="*80 + "\n")
    return timing_file

# ------------------------------------------------------------------------------
# Main Runner
# ------------------------------------------------------------------------------

def run_app(algorithm_list, algorithm_list_instances):
    args = get_args_from_cmdline()

    # Get dataset and entity from args (command line overrides, or use defaults).

    dataset = str(args.get('dataset', 'skab')).lower()
    entity = str(args.get('entity', '3'))
    use_parallel = args.get('parallel', False)
    enable_online_phase = args.get('enable_online', False)  # Online phase OFF by default
    iteration = args.get('iteration', 5)  # Default to 5 iterations for window sizing
    # Online scenario: 'all' or one scenario key (aliases already resolved by
    # Utils.pipeline_spec.resolve_online_settings).
    strategy = args.get('strategy', 'periodic')
    inject_online_regime = args.get('inject_online_regime', False)  # Regime shifts on online data only
    max_online_windows = args.get('max_online_windows', None)  # Limit online windows (None = no limit)
    explain = args.get('explain', False)  # Explainability OFF by default; --explain enables it
    stages = set(args.get('stages', ALL_STAGES))  # Which stage-6 sub-stages to run
    is_partial = stages != ALL_STAGES  # Strict subset → partial run (stop after selected stages)
    skip_gan = args.get('skip_gan', False)  # --skip_gan drops the GAN sub-stage, keeping the rest
    # Synthetic anomalies injected at stage 4: one type or several (--anomaly_type a,b).
    anomaly_types = list(args.get('anomaly_types')
                         or [args.get('anomaly_type', DEFAULT_ANOMALY_TYPE)])
    anomaly_rate = args.get('anomaly_rate')  # None = the per-type defaults in anomaly_parameters.py
    decision_metric = args.get('decision_metrics', DEFAULT_DECISION_METRICS)  # Fitness the run maximises

    # Which base detectors to select among. None (the default) means all of
    # them; only the families of the requested detectors get trained.
    requested_detectors = args.get('detectors')
    detectors_to_load = list(requested_detectors or algorithm_list_instances)
    families_to_train = families_for(detectors_to_load)

    data_dir = args['dataset_path']

    if tsb_ad.is_tsb_ad(dataset):
        # A TSB-AD series may be named by its id, its full file stem or its
        # file name; every directory this run writes (checkpoints, results)
        # uses the one id, so two spellings never split one series' results.
        entity = tsb_ad.canonical_entity(data_dir, dataset, entity)
        args['entity'] = entity

    # What the online windows are scored on, resolved before anything trains:
    # a request the data cannot honour (real labels on a dataset without
    # them, or real labels with --inject_online_regime) stops the run here
    # rather than after the offline phase. Only checked when the phase runs.
    online_label_source = None
    args['labels_available'] = False
    if enable_online_phase:
        args['labels_available'] = provides_real_labels(dataset, entity, data_dir)
        try:
            online_label_source = resolve_online_labels(
                args.get('online_labels'),
                args['labels_available'],
                inject_online_regime=inject_online_regime, dataset=dataset)
        except ValueError as e:
            logger.error(f"❌ {e}")
            return
    args['online_label_source'] = online_label_source

    logger.info("="*80)
    logger.info(f"🚀 STARTING RAMSeS EXECUTION: dataset={dataset}, entity={entity}, parallel={use_parallel}, online_phase={enable_online_phase}, iteration={iteration}, strategy={strategy}, online_regime={inject_online_regime}, online_labels={online_label_source or '-'}, max_windows={max_online_windows}, stages={','.join(sorted(stages))}, skip_gan={skip_gan}, detectors={len(detectors_to_load)}/{len(algorithm_list_instances)}")
    logger.info("="*80)
    
    logger.info("📂 STAGE 1/7: Loading Training Data...")
    train_data = load_data(
        dataset=dataset, group='train',
        entities=entity, downsampling=args['downsampling'],
        min_length=args['min_length'], root_dir=data_dir, normalize=True, verbose=False
    )
    logger.info(f"✓ Training data loaded: {len(train_data.entities)} entity(ies)")
    
    logger.info("📂 STAGE 2/7: Loading Test Data...")
    test_data = load_data(
        dataset=dataset, group='test',
        entities=entity, downsampling=args['downsampling'],
        min_length=args['min_length'], root_dir=data_dir, normalize=True, verbose=False
    )
    logger.info(f"✓ Test data loaded: {len(test_data.entities)} entity(ies)")

    if not train_data.entities:
        logger.error("Failed to load training data. Check dataset and paths.")
        return
    # Univariate-only families are dropped from the pool HERE, the first point
    # at which the channel count is known.
    #
    # Declaring the restriction is not enough on its own: nothing else consumed
    # UNIVARIATE_FAMILIES, so selecting one of these on a multivariate entity
    # reached `_TSBADEstimator._check_width`, whose ValueError is raised from
    # inside `TrainModels.train_models` — then a loop with no per-family
    # recovery, so one unusable detector aborted training for every family
    # after it. Training now survives a failing instance, but dropping them up
    # front is still the better answer: the warning names the actual reason
    # (the channel count) instead of reporting each one as a training failure,
    # and nothing is forked just to be refused.
    n_channels = test_data.entities[0].Y.shape[0]
    if n_channels > 1:
        dropped = [d for d in detectors_to_load
                   if family_of(d) in UNIVARIATE_FAMILIES]
        if dropped:
            detectors_to_load = [d for d in detectors_to_load if d not in dropped]
            families_to_train = families_for(detectors_to_load)
            logger.warning(
                f"⚠ Skipping {len(dropped)} univariate-only detector(s) on a "
                f"{n_channels}-channel entity: {', '.join(dropped)}. "
                f"Table I marks these 'U'; they run on UCR.")
            if not detectors_to_load:
                logger.error(
                    "Every requested detector is univariate-only and this "
                    f"entity has {n_channels} channels — nothing left to run. "
                    "Select a multivariate detector, or run on UCR.")
                return
    else:
        dropped = [d for d in detectors_to_load
                   if family_of(d) in MULTIVARIATE_FAMILIES]
        if dropped:
            detectors_to_load = [d for d in detectors_to_load if d not in dropped]
            families_to_train = families_for(detectors_to_load)
            logger.warning(
                f"⚠ Skipping {len(dropped)} multivariate-only detector(s) on a "
                f"1-channel entity: {', '.join(dropped)}. They score without "
                f"raising here, but the scores are not what the method measures.")
            if not detectors_to_load:
                logger.error(
                    "Every requested detector is multivariate-only and this "
                    "entity has 1 channel — nothing left to run. "
                    "Select a univariate detector, or run on SKAB or SMD.")
                return

    logger.info("🔧 STAGE 3/7: Training/Loading Models...")
    # `random_seed` from the config, which this file never applied. Runs were
    # reproducible only by accident: training in this process seeded the
    # global RNGs on its way (LSTMVAE, RNN and DGHL call torch.manual_seed,
    # the Trainer calls np.random.seed), and whatever trained after them, or
    # ran after stage 3, inherited that. With `training_timeout` set each
    # instance trains in a forked child, which inherits the RNG state this
    # process has at the fork and hands nothing back. Seeded here, every child
    # starts from the configured seed — CBLOF's KMeans, for one, draws its
    # initial centres from the global numpy RNG, and was measured to fit
    # different centres on skab/1 from one unseeded run to the next.
    set_all_seeds(args['random_seed'])
    model_trainer = TrainModels(
        dataset=dataset,
        entity=entity,
        algorithm_list=families_to_train,
        detectors=detectors_to_load,
        downsampling=args['downsampling'],
        min_length=args['min_length'],
        root_dir=args['dataset_path'],
        training_size=args['training_size'],
        overwrite=args['overwrite'],
        verbose=args['verbose'],
        save_dir=args['trained_model_path'],
        # A config written before the key existed gets the default; an explicit
        # null or 0 turns the limit off. Safe to fork here because this process
        # has run nothing but data loading yet — see Model_Training/deadline.py.
        training_timeout=args.get('training_timeout', TRAINING_TIMEOUT_SECONDS),
    )

    try:
        # Train (no-op if already present, depending on TrainModels implementation)
        logger.info("  → Checking existing models and training missing ones...")
        model_trainer.train_models(model_architectures=args['model_architectures'])
        logger.info("  ✓ Model training phase complete")

        # A detector that failed to train in THIS run is left out even when an
        # older checkpoint of it is on disk. With --overwrite true that file is
        # what the run was asked to replace, possibly written by code the
        # current scoring no longer accepts, and the failed retrain leaves it
        # untouched (checkpoints are renamed into place only once complete).
        # Loading it would quietly mix model generations; the file stays, the
        # run goes on without it, and the log says so.
        failed_to_train = [d for d in detectors_to_load if d in model_trainer.failed]
        if failed_to_train:
            logger.warning(
                f"⚠ Left out of this run because they failed to train: "
                f"{', '.join(failed_to_train)}. The warnings above give each reason.")

        # Load trained models using the dynamic path (base_dir/dataset/entity/)
        logger.info("  → Loading trained models from disk...")
        global trained_models
        models_dir = os.path.join(args['trained_model_path'], dataset, entity)
        # Probed against the test split as loaded, before injection. The probe
        # scores copies and restores every random generator it touched, so the
        # injection below draws exactly what it would have without it.
        trained_models, excluded = load_trained_models(
            [d for d in detectors_to_load if d not in model_trainer.failed],
            models_dir, probe_data=test_data)

        # Filter to only the detectors that actually loaded. A requested detector
        # with no checkpoint, or with one that cannot be loaded or cannot score
        # this data, degrades to "run without it, say so loudly" rather than
        # failing the run — but
        # below two detectors there is nothing to select between, and GA
        # fitness / rank aggregation / the off-by pairwise surrogates would all
        # be vacuous.
        loaded_model_names = list(trained_models.keys())
        missing = [d for d in detectors_to_load
                   if d not in trained_models and d not in excluded
                   and d not in model_trainer.failed]
        if missing:
            logger.warning(f"⚠ Requested detectors with no trained model (skipped): {', '.join(missing)}")
        if len(loaded_model_names) < MIN_DETECTORS:
            # Logged and returned rather than raised: the outer handler would
            # print a traceback that says nothing the sentence below does not,
            # and the web UI reports the last ERROR line as the failure reason
            # but a traceback only as "the pipeline raised an exception".
            why = ""
            if failed_to_train:
                why += (f" Failed to train in this run: "
                        f"{'; '.join(f'{d} ({model_trainer.failed[d]})' for d in failed_to_train)}.")
            slow = [d for d in excluded if d in timed_out_detectors()]
            incompatible = [d for d in excluded if d not in slow]
            if incompatible:
                why += (f" Excluded because their checkpoint cannot be loaded or "
                        f"cannot score this data: {', '.join(incompatible)} — retrain them with "
                        f"--overwrite true (the reasons are in the warnings above).")
            if slow:
                why += (f" Excluded for exceeding the scoring deadline on a short "
                        f"probe: {', '.join(slow)}.")
            if missing:
                why += (f" No trained model: {', '.join(missing)} — train them or "
                        f"widen --detectors.")
            logger.error(
                f"❌ Need at least {MIN_DETECTORS} usable detectors to run model "
                f"selection; {len(loaded_model_names)} remain "
                f"({', '.join(loaded_model_names) or 'none'}).{why}")
            return
        if len(loaded_model_names) < 3:
            logger.warning(f"⚠ Only {len(loaded_model_names)} detectors available — ensemble and "
                           f"ranking explanations will be degenerate.")
        logger.info(f"✓ Loaded {len(loaded_model_names)} models: {', '.join(loaded_model_names)}")

        # Seeded again, because what stage 3 leaves in the RNGs depends on how
        # it ran: trained here (each DL family re-seeds and then draws), trained
        # in forked children (untouched), or skipped because every checkpoint
        # was on disk (untouched). Everything from here on draws from them —
        # the injected anomaly's position and shape, LSTMVAE's latent sampling
        # in every scoring call — and the selected ensemble was measured to
        # change between two identical forked runs without this. Re-seeding
        # makes stages 4-6 start from one state in all three cases.
        set_all_seeds(args['random_seed'])

        logger.info("💉 STAGE 4/7: Injecting Synthetic Anomalies...")
        # Inject anomalies
        anomaly_list = list(anomaly_types)
        test_data_before = copy.deepcopy(test_data)
        train_data_before = copy.deepcopy(train_data)

        train_data, _ = Inject(train_data, anomaly_list, rate=anomaly_rate)
        test_data, anomaly_sizes = Inject(test_data, anomaly_list, rate=anomaly_rate)
        logger.info(f"✓ Injected anomalies: {anomaly_list}"
                    + (f" at rate {anomaly_rate}" if anomaly_rate is not None else ""))

        logger.info("📊 STAGE 5/7: Preparing Data and Visualization...")
        # Simple visualization of injected region
        anomaly_start = int(np.argmax(test_data.entities[0].labels))
        anomaly_end = test_data.entities[0].Y.shape[1] - int(np.argmax(test_data.entities[0].labels[::-1]))
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20, 6))
        axes[0].plot(test_data.entities[0].Y.flatten())
        axes[0].plot(np.arange(anomaly_start, anomaly_end),
                     test_data.entities[0].Y.flatten()[anomaly_start:anomaly_end],
                     color='red')
        axes[0].plot(np.arange(anomaly_start, anomaly_end),
                     test_data_before.entities[0].Y.flatten()[anomaly_start:anomaly_end],
                     linestyle='--')
        axes[0].set_title('Test data with Injected Anomalies', fontsize=16)
        axes[1].plot(anomaly_sizes.flatten())
        axes[1].plot(test_data.entities[0].labels.flatten(), color='red')
        axes[1].set_title('Anomaly Scores', fontsize=16)

        out_dir = results_dir("GA_Ens", dataset, entity)
        os.makedirs(out_dir, exist_ok=True)
        # The anomaly type used to be in this name, so switching type accumulated
        # variants in the gallery instead of replacing the previous run's figure.
        out_file = os.path.join(out_dir, f"ensemble_scores_{dataset}_{entity}_Data_vs_anomalies.png")
        plt.savefig(out_file, dpi=300)
        logger.info(f"✓ Visualization saved to {out_file}")

        logger.info("🔍 STAGE 6/7: Running Model Selection Algorithms...")
        logger.info("  This includes: GA, Thompson Sampling, GAN, Monte Carlo, Off-by-Threshold")
        
        # Data splitting: 80% for offline phase, 20% for online phase (as per
        # paper). The split, the window geometry and the feasibility verdict
        # come from Utils.pipeline_spec, which the web UI's feasibility check
        # calls too, so the two cannot disagree.
        data = test_data_before.entities[0].Y
        targets = test_data_before.entities[0].labels
        mask = test_data_before.entities[0].mask
        total_samples = int(np.size(targets.flatten()))
        offline_size, online_size = online_split(total_samples)
        # With real labels the verdict also needs a window that can be scored
        # on them: an online part without a labelled anomaly has nothing to
        # score, and the phase is skipped with that reason.
        feasibility = online_feasibility(
            total_samples, iteration,
            online_labels=(targets.flatten()[offline_size:]
                           if online_label_source == ONLINE_LABELS_REAL else None))
        
        # For offline phase: use first 80% of data
        offline_data = data[:, :offline_size]
        offline_targets = targets.flatten()[:offline_size].reshape(1, -1) if targets.ndim > 1 else targets[:offline_size]
        offline_mask = mask[:, :offline_size]
        
        # For online phase: the remaining 20%, cut into sliding windows later
        online_data = data[:, offline_size:]
        online_targets = targets.flatten()[offline_size:].reshape(1, -1) if targets.ndim > 1 else targets[offline_size:]
        online_mask = mask[:, offline_size:]
        
        # Inject regime shifts into ONLINE data only (if requested)
        if inject_online_regime:
            logger.info(f"  ⚠️  INJECTING REGIME SHIFTS into online data (scale + wander)...")
            from Datasets.dataset import Entity, Dataset
            # Create temporary dataset for injection
            online_entity = Entity(Y=online_data, name=f"{entity}_online", labels=online_targets, verbose=False)
            online_dataset = Dataset(entities=[online_entity], name=f"{dataset}_online", verbose=False)
            # Inject regime shifts (scale + wander). Seeded, with numpy's global
            # generator restored afterwards, so the shifted series is the same
            # on every run and the offline phase's randomness is untouched.
            _rng_state = np.random.get_state()
            np.random.seed(int(13 if args.get('random_seed') is None else args['random_seed']))
            try:
                online_dataset, regime_sizes = Inject(online_dataset, list(REGIME_SHIFT_TYPES))
            finally:
                np.random.set_state(_rng_state)
            # Extract modified data. Inject concatenates one copy of the series
            # per parameter combination, so the mask must be taken from it too:
            # the original one no longer matched the data's length. The online
            # stream is that concatenation, and window positions count in it.
            online_data = online_dataset.entities[0].Y
            online_targets = online_dataset.entities[0].labels
            online_mask = online_dataset.entities[0].mask
            logger.info(f"  ✓ Regime shifts injected: scale + wander "
                        f"({online_data.shape[1]} online samples after injection)")
        
        # Online phase: sliding windows over the online stream. window_size =
        # online_size / iteration (at least 100 samples), step_size = 5% of the
        # window (at least 5), i.e. 95% overlap between consecutive windows.
        window_size, step_size = feasibility['window_size'], feasibility['step_size']
        num_windows = count_windows(online_data.shape[1], window_size, step_size)
        
        logger.info(f"  📊 Data split: Total={total_samples}, Offline={offline_size} (80%), Online={online_size} (20%)")
        logger.info(f"  📊 Online windows: window_size={window_size} (online_size/{iteration}), step_size={step_size} (5% of window)")
        logger.info(f"  📊 {num_windows} sliding windows fit the online stream"
                    + ("" if feasibility['ok'] else f" — {feasibility['reason']}"))
        
        # ==================== OFFLINE PHASE ====================
        logger.info("  🔧 OFFLINE PHASE: Running full model selection on 80% of data...")
        
        # Use offline data for initial model selection
        test_data.entities[0].Y = offline_data
        test_data.entities[0].labels = offline_targets
        test_data.entities[0].mask = offline_mask
        test_data.entities[0].n_time = offline_size
        test_data.total_time = offline_size

        # The robustness stages (GAN, off-by-threshold, Monte Carlo) score the
        # offline part WITHOUT the synthetic anomalies. They used to receive
        # the whole test series, online 20% included: future data leaking into
        # the offline selection.
        offline_raw_data = copy.deepcopy(test_data)

        test_data_new = copy.deepcopy(test_data)
        logger.info(f"  📏 Before Inject: test_data size = {test_data.entities[0].labels.shape[1] if test_data.entities[0].labels.ndim > 1 else test_data.entities[0].labels.shape[0]}")
        # The copies are kept for the online phase: a re-optimisation slides
        # each injected copy forward in time on its own.
        test_data_new, _, offline_copies = Inject(test_data_new, anomaly_list,
                                                  rate=anomaly_rate, return_copies=True)

        # Log data sizes for architecture verification
        offline_raw_size = offline_raw_data.entities[0].labels.shape[1] if offline_raw_data.entities[0].labels.ndim > 1 else offline_raw_data.entities[0].labels.shape[0]
        test_data_new_size = test_data_new.entities[0].labels.shape[1] if test_data_new.entities[0].labels.ndim > 1 else test_data_new.entities[0].labels.shape[0]
        logger.info(f"  📏 After Inject: offline raw (GAN, off-by, Monte Carlo) = {offline_raw_size}, test_data_new (injected, for GA and Thompson) = {test_data_new_size}")

        # Save a clean copy of injected test data BEFORE model selection mutates it
        # (Thompson Sampling processes it window-by-window, leaving only the last window)
        test_data_new_for_eval = copy.deepcopy(test_data_new)

        # Start end-to-end timing
        e2e_start_time = time.time()
        # Partial runs (a strict subset of stages) always take the sequential path
        # and stop after the selected stages — see the early-exit just below.
        if use_parallel and not is_partial:
            logger.info("  ⏱ Starting model selection pipeline (PARALLEL mode)...")
            (best_thompson, robust_agg, full_aggregated, best_ensemble,
             individual_predictions, base_model_predictions_train, base_model_predictions_test,
             y_true_train, y_true_test, meta_model_type, extra_results) = run_model_selection_algorithms_2(
                train_data, test_data_new, dataset, entity, iteration=OFFLINE_ITERATION,
                trained_models=trained_models, model_list=loaded_model_names,
                test_data_gan=offline_raw_data, skip_gan=skip_gan, explain=explain,
                decision_metric=decision_metric
            )
        else:
            mode = f"PARTIAL: {','.join(sorted(stages))}" if is_partial else "SEQUENTIAL mode"
            logger.info(f"  ⏱ Starting model selection pipeline ({mode})...")
            (best_thompson, robust_agg, full_aggregated, best_ensemble,
             individual_predictions, base_model_predictions_train, base_model_predictions_test,
             y_true_train, y_true_test, meta_model_type, extra_results) = run_model_selection_algorithms_1(
                train_data, test_data_new, dataset, entity, iteration=OFFLINE_ITERATION,
                model_list=loaded_model_names, test_data_gan=offline_raw_data,
                skip_gan=skip_gan, explain=explain,
                stages=stages, decision_metric=decision_metric
            )

        # Calculate end-to-end time
        e2e_time = time.time() - e2e_start_time

        # Partial run: the selected stages (+ their explainability) are done. The
        # downstream rank aggregation, ensemble-vs-single decision, comprehensive
        # report and online phase all require the full set of stages, so stop here.
        if is_partial:
            logger.info("="*80)
            logger.info(f"✅ Partial run complete (stages={','.join(sorted(stages))}). "
                        f"Skipped rank aggregation, final decision, comprehensive report & online phase. "
                        f"Time: {e2e_time:.2f}s")
            logger.info("="*80)
            return
        logger.info(f"✓ Model selection completed in {e2e_time:.2f}s ({e2e_time/60:.2f} min)")
        
        # Helper function to convert any value to scalar
        def to_scalar(value):
            """Convert tuple, list, array, or scalar to a single float value."""
            # Handle nested structures
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return 0.0
                value = value[0]
            if isinstance(value, np.ndarray):
                value = float(value.flat[0]) if value.size > 0 else 0.0
            value = float(value)
            # Validate F1/PR-AUC is in valid range [0, 1]
            if value < 0.0 or value > 1.0:
                logger.warning(f"Invalid F1/PR-AUC score detected: {value}. This should be between 0 and 1.")
            return value
        
        # Decide: Ensemble vs Single Model based on scores
        ensemble_f1 = to_scalar(extra_results['ga']['f1'])
        ensemble_pr_auc = to_scalar(extra_results['ga']['pr_auc'])
        
        # Get best single model from final aggregation (top of the flat ranking list)
        best_single_model = full_aggregated[0] if isinstance(full_aggregated, (list, tuple)) and len(full_aggregated) > 0 else full_aggregated
        
        # Evaluate Thompson Sampling best model on the SAME injected data the ensemble sees
        # (test_data_new). This keeps single-model and ensemble F1 directly comparable.
        best_thompson_model = best_thompson if isinstance(best_thompson, str) else (best_thompson[0] if best_thompson else 'N/A')
        test_data_for_thompson = copy.deepcopy(test_data_new_for_eval)
        _, thompson_scores, thompson_f1_list, thompson_pr_list = evaluate_individual_models(
            [best_thompson_model], test_data_for_thompson, trained_models
        )
        thompson_f1 = to_scalar(thompson_f1_list[0]) if len(thompson_f1_list) > 0 else 0.0
        thompson_pr_auc = to_scalar(thompson_pr_list[0]) if len(thompson_pr_list) > 0 else 0.0
        
        # Evaluate the best single model from final aggregation on the SAME injected data
        test_data_for_eval = copy.deepcopy(test_data_new_for_eval)
        single_preds, _, single_f1_list, single_pr_list = evaluate_individual_models(
            [best_single_model], test_data_for_eval, trained_models
        )
        single_model_f1 = to_scalar(single_f1_list[0]) if len(single_f1_list) > 0 else 0.0
        single_model_pr_auc = to_scalar(single_pr_list[0]) if len(single_pr_list) > 0 else 0.0

        # One window for both candidates, so their VUS values compare. The
        # predictions dict carries the continuous scores VUS needs.
        vus_win = vus_window(test_data_new_for_eval.entities[0].Y)
        single_true, single_raw = single_preds.get(best_single_model, (None, None))
        single_model_vus = (vus_score(single_raw, single_true, vus_win)
                            if single_raw is not None else float("nan"))
        ensemble_vus = vus_score(extra_results['ga'].get('scores'),
                                 extra_results['ga'].get('y_true'), vus_win)
        
        # Evaluate Monte Carlo best models on the SAME injected data (Option B: symmetric eval)
        best_mc_f1_model = extra_results['monte_carlo'].get('best_model_f1', 'N/A')
        best_mc_pr_model = extra_results['monte_carlo'].get('best_model_pr_auc', 'N/A')
        
        mc_f1_score = 0.0
        mc_pr_auc_score = 0.0
        if best_mc_f1_model != 'N/A':
            test_data_for_mc = copy.deepcopy(test_data_new_for_eval)
            _, _, mc_f1_list, _ = evaluate_individual_models([best_mc_f1_model], test_data_for_mc, trained_models)
            mc_f1_score = to_scalar(mc_f1_list[0]) if len(mc_f1_list) > 0 else 0.0
        
        if best_mc_pr_model != 'N/A':
            test_data_for_mc2 = copy.deepcopy(test_data_new_for_eval)
            _, _, _, mc_pr_list = evaluate_individual_models([best_mc_pr_model], test_data_for_mc2, trained_models)
            mc_pr_auc_score = to_scalar(mc_pr_list[0]) if len(mc_pr_list) > 0 else 0.0
        
        # Framework decision, on whichever metric --decision_metric names. Ties go
        # to the ensemble, as they did when this was F1-only.
        ensemble_metrics = {'f1': ensemble_f1, 'pr_auc': ensemble_pr_auc, 'vus': ensemble_vus}
        single_metrics = {'f1': single_model_f1, 'pr_auc': single_model_pr_auc, 'vus': single_model_vus}
        # A metric neither side could produce (VUS on a short window) drops out
        # and the remaining weights are renormalised, so the formula the report
        # prints is the one the comparison was actually made on.
        wanted = metrics_required(decision_metric)
        available = tuple(m for m in wanted
                          if not (np.isnan(ensemble_metrics[m]) or np.isnan(single_metrics[m])))
        if available != wanted:
            decision_metric = (restrict_metrics(decision_metric, available)
                               if available else DEFAULT_DECISION_METRICS)
            logger.warning(f"{decision_metric_label(wanted)} unavailable for the final "
                           f"decision; deciding on {decision_metric_formula(decision_metric)}")
        ensemble_score = combine_metrics(decision_metric, ensemble_metrics)
        single_model_score = combine_metrics(decision_metric, single_metrics)
        framework_choice = 'ensemble' if ensemble_score >= single_model_score else 'single_model'
        
        # Prepare comprehensive results using captured data
        results_dict = {
            'ga': {
                'ensemble': best_ensemble,
                'f1': ensemble_f1,
                'pr_auc': ensemble_pr_auc,
                'fitness': extra_results['ga']['fitness'],
                'meta_model_type': meta_model_type,
                'chosen_model': best_ensemble
            },
            'thompson': {
                'top_models': extra_results['thompson'],
                'best_model': best_thompson_model,
                'f1': thompson_f1,
                'pr_auc': thompson_pr_auc
            },
            'gan_robustness': {
                'f1_names': extra_results['gan']['f1_names'],
                'pr_auc_names': extra_results['gan']['pr_auc_names'],
                'best_model': extra_results['gan'].get('best_model', 'N/A'),
                'best_f1': extra_results['gan'].get('best_f1', 0.0),
                'best_pr_auc': extra_results['gan'].get('best_pr_auc', 0.0)
            },
            'borderline': {
                'f1_names': extra_results['borderline']['f1_names'],
                'pr_auc_names': extra_results['borderline']['pr_auc_names'],
                'best_model': extra_results['borderline'].get('best_model', 'N/A'),
                'best_f1': extra_results['borderline'].get('best_f1', 0.0),
                'best_pr_auc': extra_results['borderline'].get('best_pr_auc', 0.0)
            },
            'monte_carlo': {
                'f1_names': extra_results['monte_carlo']['f1_names'],
                'pr_auc_names': extra_results['monte_carlo']['pr_auc_names'],
                'best_model_f1': extra_results['monte_carlo'].get('best_model_f1', 'N/A'),
                'best_model_pr_auc': extra_results['monte_carlo'].get('best_model_pr_auc', 'N/A'),
                'best_f1': mc_f1_score,
                'best_pr_auc': mc_pr_auc_score
            },
            'aggregation': {
                'robust_agg': robust_agg,
                'final_agg': full_aggregated
            },
            'final_decision': {
                'single_model': best_single_model,
                'ensemble': best_ensemble,
                'meta_model_type': meta_model_type,
                'single_model_f1': single_model_f1,
                'single_model_pr_auc': single_model_pr_auc,
                'single_model_vus': single_model_vus,
                'ensemble_vus': ensemble_vus,
                'ensemble_f1': ensemble_f1,
                'ensemble_pr_auc': ensemble_pr_auc,
                'ensemble_fitness': extra_results['ga']['fitness'],
                'decision_metric': decision_metric,
                'ensemble_score': ensemble_score,
                'single_model_score': single_model_score,
                'framework_choice': framework_choice,
                'chosen_model': best_ensemble if framework_choice == 'ensemble' else best_single_model
            }
        }
        
        timing_dict = {
            'modules': extra_results.get('timing', {}),
            'total': e2e_time
        }
        
        # Get memory dict from extra_results
        memory_dict = extra_results.get('memory', None)
        
        # Global Intermediate Representation: combine the per-stage IR JSONs with
        # the decision context (grounded LLM input; non-fatal). Runs BEFORE the
        # comprehensive results are written so the LLM layer's runtime and
        # memory land in the report like every other pipeline module.
        if explain:
            explain_mem_before = get_memory_usage_mb()
            explain_start = time.time()
            try:
                from Explainability.ir import assemble_global_ir
                ir_path = assemble_global_ir(results_dict, dataset, entity, OFFLINE_ITERATION)
                logger.info(f"✓ Global explanation IR written to: {ir_path}")
            except Exception as e:
                logger.error(f"Global IR assembly failed (non-fatal): {e}")

            # LLM narration over the IR files (non-fatal; skipped with a hint when
            # no local LLM server is reachable). Regenerable any time via
            # `python -m Explainability.narrate`.
            try:
                from Explainability.llm import (DEFAULT_BASE_URL, DEFAULT_MODEL,
                                                LLMClient, narrate_entity)
                llm_client = LLMClient(
                    base_url=args.get('llm_base_url') or DEFAULT_BASE_URL,
                    model=args.get('llm_model') or DEFAULT_MODEL)
                logger.info(f"📝 Generating LLM narratives ({llm_client.model})...")
                nl_report = narrate_entity(dataset, entity, OFFLINE_ITERATION, llm_client)
                ov = nl_report['overall']
                logger.info(f"✓ LLM narratives written "
                            f"(hallucination {ov['hallucination_rate']:.3f}, "
                            f"omission {ov['omission_rate']:.3f}): "
                            f"{nl_report['faithfulness_txt']}")
            except ConnectionError as e:
                logger.warning(f"LLM narration skipped — {e}")
            except Exception as e:
                logger.error(f"LLM narration failed (non-fatal): {e}")

            explain_mem_after = get_memory_usage_mb()
            timing_dict['modules']['7_LLM_Explainability'] = time.time() - explain_start
            if memory_dict is not None:
                memory_dict.setdefault('modules', {})['7_LLM_Explainability'] = {
                    'before': explain_mem_before,
                    'after': explain_mem_after,
                    'delta': explain_mem_after - explain_mem_before,
                    'peak': _MEM_TRACE.peak_between(explain_start, time.time()),
                    # The model itself lives in the LLM server process, not
                    # here. Without this the stage reads as ~20 MB.
                    'llm_server_peak': _MEM_TRACE.llm_peak_between(
                        explain_start, time.time()),
                }
                # The narration ran after the algorithms' final/peak snapshot.
                # Keep that reading — it is the undistorted end of the compute
                # pipeline — and record the post-narration one beside it, where
                # the process has been paged out while blocked on the server.
                memory_dict['final_algorithms'] = memory_dict.get('final')
                memory_dict['final'] = get_memory_usage_mb()
                memory_dict['peak'] = get_peak_memory_mb()
            e2e_time = time.time() - e2e_start_time
            timing_dict['total'] = e2e_time

        logger.info("📝 STAGE 7/7: Writing Comprehensive Results...")
        # Write comprehensive results
        comp_results_dir = results_dir("comprehensive", dataset, entity)
        os.makedirs(comp_results_dir, exist_ok=True)
        comp_results_file = os.path.join(
            comp_results_dir, f"comprehensive_results_{dataset}_{entity}_iter{iteration}.txt"
        )
        _MEM_TRACE.stop()
        write_comprehensive_results(comp_results_file, dataset, entity, iteration, results_dict, timing_dict, memory_dict)
        logger.info(f"✓ Results written to: {comp_results_file}")
        logger.info("="*80)
        logger.info(f"🎉 EXECUTION COMPLETE! Total Time: {e2e_time:.4f}s ({e2e_time/60:.2f} min)")
        logger.info("="*80)

        # ==================== ONLINE PHASE ====================
        # The remaining 20% of the series, window by window, under each
        # requested re-optimisation scenario (see run_online_phase).

        # Save initial offline phase selection (window 0)
        save_current_selection(
            dataset, entity, window_idx=0, best_ensemble=best_ensemble,
            best_single_model=best_single_model,
            meta_model_type=meta_model_type, trained_models=trained_models,
            individual_predictions=individual_predictions,
            base_model_predictions_train=base_model_predictions_train,
            base_model_predictions_test=base_model_predictions_test,
            y_true_train=y_true_train, y_true_test=y_true_test
        )

        if not enable_online_phase:
            logger.info(f"  ℹ️  Online phase disabled (20% of data reserved but not processed)")
            logger.info(f"  ℹ️  To enable online phase, add --enable_online flag")
        else:
            run_online_phase(
                args=args, dataset=dataset, entity=entity, train_data=train_data,
                trained_models=trained_models, loaded_model_names=loaded_model_names,
                models_dir=models_dir, offline_copies=offline_copies,
                offline_raw=(offline_data, np.asarray(offline_targets).reshape(-1),
                             offline_mask),
                stream_raw=(online_data, np.asarray(online_targets).reshape(-1),
                            online_mask),
                feasibility=feasibility, window_size=window_size, step_size=step_size,
                total_samples=total_samples, offline_size=offline_size,
                online_size=online_size, anomaly_list=anomaly_list,
                anomaly_rate=anomaly_rate, best_ensemble=best_ensemble,
                best_single_model=best_single_model, meta_model_type=meta_model_type,
                ga_meta_model=extra_results['ga'].get('meta_model'),
                individual_predictions=individual_predictions,
                base_model_predictions_train=base_model_predictions_train,
                y_true_train=y_true_train,
                offline_summary={
                    'framework_choice': framework_choice,
                    'decision_formula': decision_metric_formula(decision_metric),
                    'ensemble': {'members': list(best_ensemble or []),
                                 'meta_model_type': meta_model_type,
                                 'scores': {'f1': ensemble_f1, 'pr_auc': ensemble_pr_auc,
                                            'vus': ensemble_vus, 'score': ensemble_score}},
                    'single': {'model': best_single_model,
                               'scores': {'f1': single_model_f1,
                                          'pr_auc': single_model_pr_auc,
                                          'vus': single_model_vus,
                                          'score': single_model_score}},
                    # The range-based metrics' tolerance grows with the series
                    # (about a tenth of it), so these offline scores are on a
                    # different scale from any window's.
                    'comparable_with_windows': False,
                    'comparability_note': (
                        'Offline scores were measured on the whole offline series, '
                        'window scores on 100-sample windows; the range-based F1 and '
                        'PR-AUC tolerance grows with the series length, so the two '
                        'are not on one scale. Each window signal is compared only '
                        'with its own online baseline.'),
                },
                vus_win=vus_win, iteration=iteration, offline_time=e2e_time,
            )

    except Exception:
        logger.info('Traceback for Entity: %s Dataset: %s', entity, dataset)
        logger.error(traceback.format_exc())

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Global so the selection functions can access after initial load
    trained_models = {}
    run_app(algorithm_list, algorithm_list_instances)
