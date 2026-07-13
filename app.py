# time_series_framework/app.py
import concurrent.futures
import copy
import logging
import os
import psutil
import time
import traceback
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch as t

from Datasets.load import load_data
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
from Model_Selection.Thompson_Sampling import (
    run_linear_thompson_sampling,
    initialize_sliding_windows,
)
from Model_Selection.inject_anomalies import Inject
from Model_Selection.rank_aggregation import (
    enhanced_markov_chain_rank_aggregator_text,
    explain_rank_aggregation,
)
from Model_Training.train import TrainModels
from Utils.utils import get_args_from_cmdline
# from comprehensive_results_writer import write_comprehensive_results

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# save_dir removed - now dynamically determined from command-line args
# Old hardcoded path was causing wrong models to load for different datasets

algorithm_list = ['NN', 'LOF', 'CBLOF']
algorithm_list_instances = [
        'LOF_1', 'LOF_2', 'LOF_3', 'LOF_4', 'NN_1', 'NN_2', 'NN_3',
        'CBLOF_1', 'CBLOF_2', 'CBLOF_3', 'CBLOF_4'
    ]

# Sub-stages of the offline model-selection pipeline (stage 6). The --stages CLI
# flag selects a subset; a strict subset is a "partial run" (runs only those
# stages + their explainability, then stops before rank aggregation).
ALL_STAGES = {"ga", "thompson", "gan", "offby", "montecarlo"}

# Iteration tag of the OFFLINE model-selection run. The offline pipeline and its
# artifacts (rank-aggregation reports, explanation IRs, narration) all use this
# value; the CLI --iteration arg is the window-sizing parameter, a different
# concept that must not leak into these filenames.
OFFLINE_ITERATION = 0

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
    """Get peak memory usage in MB (platform-dependent)."""
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # macOS: ru_maxrss is in bytes, Linux: in kilobytes
        if os.uname().sysname == 'Darwin':
            return rusage.ru_maxrss / (1024 * 1024)
        else:
            return rusage.ru_maxrss / 1024
    except:
        # Fallback to current usage if resource module unavailable
        return get_memory_usage_mb()

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
        f.write(f"Dataset: {dataset} | Entity: {entity} | Iteration: {iteration}\n")
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
            f.write("-" * 50 + "\n")
            for module, mem_info in memory_dict.get('modules', {}).items():
                f.write(f"  {module:<30} : {mem_info['after']:>10.2f} MB\n")
                f.write(f"    Delta: +{mem_info['delta']:>10.2f} MB\n")
            
            f.write("-" * 50 + "\n")
            f.write(f"  {'Initial Memory':<30} : {memory_dict.get('initial', 0):>10.2f} MB\n")
            f.write(f"  {'Final Memory':<30} : {memory_dict.get('final', 0):>10.2f} MB\n")
            f.write(f"  {'Peak Memory':<30} : {memory_dict.get('peak', 0):>10.2f} MB\n")
            f.write(f"  {'Total Increase':<30} : {memory_dict.get('final', 0) - memory_dict.get('initial', 0):>10.2f} MB\n\n")
        
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
        f.write(f"  PR-AUC      : {final_decision.get('single_model_pr_auc', 0):.6f}\n\n")
        
        f.write("Ensemble Option:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Models      : {final_decision.get('ensemble', 'N/A')}\n")
        f.write(f"  Size        : {len(final_decision.get('ensemble', []))}\n")
        f.write(f"  Meta-Model  : {final_decision.get('meta_model_type', 'N/A')}\n")
        f.write(f"  F1 Score    : {final_decision.get('ensemble_f1', 0):.6f}\n")
        f.write(f"  PR-AUC      : {final_decision.get('ensemble_pr_auc', 0):.6f}\n")
        f.write(f"  Fitness     : {final_decision.get('ensemble_fitness', 0):.6f}\n\n")
        
        # Make the choice based on F1 score
        ensemble_f1 = final_decision.get('ensemble_f1', 0)
        single_f1 = final_decision.get('single_model_f1', 0)
        
        f.write("Final Choice:\n")
        f.write("-" * 50 + "\n")
        if ensemble_f1 >= single_f1:
            f.write(f"  ✓ ENSEMBLE SELECTED\n")
            f.write(f"    Reason: Ensemble F1 ({ensemble_f1:.6f}) >= Single Model F1 ({single_f1:.6f})\n")
            if ensemble_f1 > single_f1:
                improvement = ensemble_f1 - single_f1
                f.write(f"    Improvement: +{improvement:.6f} ({improvement/max(single_f1, 0.0001)*100:.2f}%)\n")
            else:
                f.write(f"    Note: F1 scores are equal\n")
        else:
            f.write(f"  ✓ SINGLE MODEL SELECTED\n")
            f.write(f"    Reason: Single Model F1 ({single_f1:.6f}) > Ensemble F1 ({ensemble_f1:.6f})\n")
            advantage = single_f1 - ensemble_f1
            f.write(f"    Advantage: +{advantage:.6f} ({advantage/max(ensemble_f1, 0.0001)*100:.2f}%)\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    logger.info(f"Comprehensive results saved to: {output_file}")


def load_trained_models(model_names, models_dir):
    """
    Load trained models from disk.

    Parameters
    ----------
    model_names : list[str]
        List of model instance names (e.g., 'CBLOF_1', 'LOF_3', ...).
    models_dir : str
        Directory where .pth files are stored.

    Returns
    -------
    dict[str, torch.nn.Module]
    """
    trained = {}
    missing = []
    for name in model_names:
        path = os.path.join(models_dir, f"{name}.pth")
        if not os.path.exists(path):
            missing.append(name)
            logger.warning(f"Model {name} not found in {models_dir}, skipping")
            continue
        with open(path, 'rb') as fh:
            model = t.load(fh, weights_only=False)
            try:
                model.eval()
            except AttributeError:
                # If the checkpoint isn't a nn.Module, we still store it as-is.
                pass
            trained[name] = model
    
    logger.info("Loaded %d trained models from %s", len(trained), models_dir)
    if missing:
        logger.warning(f"Missing {len(missing)} models: {', '.join(missing)}")
    return trained

# ------------------------------------------------------------------------------
# Model-Selection Pipelines
# ------------------------------------------------------------------------------

def run_model_selection_algorithms_1(train_data, test_data, dataset, entity, iteration, model_list=None, test_data_gan=None, skip_gan=False, explain=False, stages=None):
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
    thompson_model_names = []
    Gan_ranked_by_f1 = Gan_ranked_by_pr_auc = []
    Gan_ranked_by_f1_names = Gan_ranked_by_pr_auc_names = []
    ranked_by_f1 = ranked_by_pr_auc = []
    ranked_by_f1_names_sensitivity = ranked_by_pr_auc_names_sensitivity = []
    monte_carlo_ranked_models_F1 = monte_carlo_ranked_models_PR = []
    robust_agg = [None, []]
    full_aggregated = [None, []]

    # Track initial memory
    initial_memory = get_memory_usage_mb()
    memory_dict['initial'] = initial_memory
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
        y_true_train, y_true_test, meta_model_type = genetic_algorithm(
            dataset, entity, train_data, test_data,
            models_to_use, trained_models,
            population_size=20, generations=20,
            meta_model_type='rf', mutation_rate=0.1,
            explain=explain,
        )
        timing_dict['1_Genetic_Algorithm'] = time.time() - start_time
        mem_after = get_memory_usage_mb()
        memory_dict['modules']['1_Genetic_Algorithm'] = {
            'before': mem_before,
            'after': mem_after,
            'delta': mem_after - mem_before
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
            model_names=algorithm_list_instances,
            dataset=dataset,
            entity=entity,
            iterations=50,
            iteration=iteration,
            explain=explain,
        )
        timing_dict['2_Thompson_Sampling'] = time.time() - start_time
        mem_after = get_memory_usage_mb()
        memory_dict['modules']['2_Thompson_Sampling'] = {
            'before': mem_before,
            'after': mem_after,
            'delta': mem_after - mem_before
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
        gan_results = run_Gan(
            test_data_for_gan, trained_models, algorithm_list_instances, dataset, entity
        )
        timing_dict['3_GAN_Robustness'] = time.time() - start_time
        mem_after = get_memory_usage_mb()
        memory_dict['modules']['3_GAN_Robustness'] = {
            'before': mem_before,
            'after': mem_after,
            'delta': mem_after - mem_before
        }
        Gan_ranked_by_f1, Gan_ranked_by_pr_auc, \
        Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names = (
            gan_results[0], gan_results[1], gan_results[2], gan_results[3]
        )
        logger.info("  ✓ [GAN] F1 names top-5: %s | Time=%.4fs", Gan_ranked_by_f1_names[:5],
                    timing_dict['3_GAN_Robustness'])
        logger.info("     [GAN] PR names top-5: %s", Gan_ranked_by_pr_auc_names[:5])
    else:
        reason = "skip_gan=True" if ("gan" in stages and skip_gan) else "not in --stages"
        logger.info(f"  ⏩ Sub-stage 6.3: GAN Robustness Testing SKIPPED ({reason})")
        # Empty placeholders already set above for skipped GAN
        timing_dict['3_GAN_Robustness'] = 0.0

    # --------------------------------------------
    # 4) Off-by-threshold (borderline sensitivity)
    # --------------------------------------------
    if "offby" in stages:
        logger.info("  📊 Sub-stage 6.4: Off-by-Threshold Testing...")
        mem_before = get_memory_usage_mb()
        start_time = time.time()
        # Use original un-injected data so synthetic spike labels don't cause single-class skips
        test_data_for_borderline = copy.deepcopy(test_data_gan if test_data_gan is not None else test_data)
        ranked_by_f1, ranked_by_pr_auc, \
        ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity = run_off_by_threshold(
            test_data_for_borderline, trained_models, algorithm_list_instances, dataset, entity, explain=explain
        )
        timing_dict['4_Borderline_Sensitivity'] = time.time() - start_time
        mem_after = get_memory_usage_mb()
        memory_dict['modules']['4_Borderline_Sensitivity'] = {
            'before': mem_before,
            'after': mem_after,
            'delta': mem_after - mem_before
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
        monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR = run_monte_carlo_simulation(
            test_data_for_mc, trained_models, algorithm_list_instances, dataset, entity,
            n_simulations=2, noise_level=0.1, explain=explain,
        )
        timing_dict['5_Monte_Carlo'] = time.time() - start_time
        mem_after = get_memory_usage_mb()
        memory_dict['modules']['5_Monte_Carlo'] = {
            'before': mem_before,
            'after': mem_after,
            'delta': mem_after - mem_before
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
                'ga': {'f1': best_f1, 'pr_auc': best_pr_auc, 'fitness': best_fitness},
                'thompson': thompson_model_names,
                'gan': {
                    'f1_names': Gan_ranked_by_f1_names,
                    'pr_auc_names': Gan_ranked_by_pr_auc_names,
                    'f1_scores': Gan_ranked_by_f1,
                    'pr_auc_scores': Gan_ranked_by_pr_auc,
                    'best_model': Gan_ranked_by_f1_names[0] if len(Gan_ranked_by_f1_names) > 0 else 'N/A',
                    'best_f1': Gan_ranked_by_f1[0][1][0]['f1'] if len(Gan_ranked_by_f1) > 0 else 0.0,
                    'best_pr_auc': Gan_ranked_by_pr_auc[0][1][0]['pr_auc'] if len(Gan_ranked_by_pr_auc) > 0 else 0.0,
                },
                'borderline': {
                    'f1_names': ranked_by_f1_names_sensitivity,
                    'pr_auc_names': ranked_by_pr_auc_names_sensitivity,
                    'f1_scores': ranked_by_f1,
                    'pr_auc_scores': ranked_by_pr_auc,
                    'best_model': ranked_by_f1_names_sensitivity[0] if len(ranked_by_f1_names_sensitivity) > 0 else 'N/A',
                    'best_f1': ranked_by_f1[0][1][0]['f1'] if len(ranked_by_f1) > 0 else 0.0,
                    'best_pr_auc': ranked_by_pr_auc[0][1][0]['pr_auc'] if len(ranked_by_pr_auc) > 0 else 0.0,
                },
                'monte_carlo': {
                    'f1_names': monte_carlo_ranked_models_F1,
                    'pr_auc_names': monte_carlo_ranked_models_PR,
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
    # Robust-only aggregation in the requested order: GAN → Borderline → Monte Carlo
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
        'delta': mem_after - mem_before
    }
    logger.info("  ✓ [Aggregation] Time=%.4fs", timing_dict['6_Rank_Aggregation'])
    
    # Track final memory and peak
    memory_dict['final'] = get_memory_usage_mb()
    memory_dict['peak'] = get_peak_memory_mb()
    logger.info(f"  💾 Final memory usage: {memory_dict['final']:.2f} MB (Peak: {memory_dict['peak']:.2f} MB)")

    # -----------------------
    # Persist a concise report
    # -----------------------
    directory = f"myresults/robust_aggregated/{dataset}/{entity}/"
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


def run_model_selection_algorithms_2(train_data, test_data, dataset, entity, iteration, trained_models, model_list=None, test_data_gan=None, skip_gan=False, explain=False):
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
    
    # Track initial memory
    initial_memory = get_memory_usage_mb()
    memory_dict['initial'] = initial_memory
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
            explain=explain
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
            gan_future = executor.submit(
                run_Gan,
                test_data_gan_copy, trained_models, models_to_use,
                dataset, entity
            )
        
        borderline_future = executor.submit(
            run_off_by_threshold,
            test_data_borderline, trained_models, models_to_use,
            dataset, entity, explain=explain
        )
        monte_carlo_future = executor.submit(
            run_monte_carlo_simulation,
            test_data_montecarlo, trained_models, models_to_use,
            dataset, entity, 2, 0.1, explain=explain
        )

        # Collect results
        logger.info("     ⏳ Waiting for parallel tasks...")
        
        best_ensemble, best_f1, best_pr_auc, best_fitness, \
        individual_predictions, base_model_predictions_train, base_model_predictions_test, \
        y_true_train, y_true_test, meta_model_type = ga_future.result()
        timing_dict['1_GA'] = time.time() - overall_start
        mem_ga = get_memory_usage_mb()
        memory_dict['modules']['1_GA'] = {
            'before': initial_memory,
            'after': mem_ga,
            'delta': mem_ga - initial_memory
        }
        logger.info("     ✓ GA: ensemble=%s | F1=%.4f | PR-AUC=%.4f", best_ensemble, best_f1, best_pr_auc)
        
        thompson_model_names = thompson_future.result()
        timing_dict['2_Thompson'] = time.time() - overall_start
        mem_thompson = get_memory_usage_mb()
        memory_dict['modules']['2_Thompson'] = {
            'before': initial_memory,
            'after': mem_thompson,
            'delta': mem_thompson - initial_memory
        }
        logger.info("     ✓ Thompson: top-5=%s", thompson_model_names[:5])
        
        if not skip_gan:
            Gan_ranked_by_f1, Gan_ranked_by_pr_auc, \
            Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names = gan_future.result()
            timing_dict['3_GAN'] = time.time() - overall_start
            mem_gan = get_memory_usage_mb()
            memory_dict['modules']['3_GAN'] = {
                'before': initial_memory,
                'after': mem_gan,
                'delta': mem_gan - initial_memory
            }
            logger.info("     ✓ GAN: F1 top-5=%s", Gan_ranked_by_f1_names[:5])
        else:
            logger.info("     ⏩ GAN: SKIPPED (skip_gan=True)")
            Gan_ranked_by_f1 = []
            Gan_ranked_by_pr_auc = []
            Gan_ranked_by_f1_names = []
            Gan_ranked_by_pr_auc_names = []
            timing_dict['3_GAN'] = 0.0
        
        ranked_by_f1, ranked_by_pr_auc, \
        ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity = borderline_future.result()
        timing_dict['4_Borderline'] = time.time() - overall_start
        mem_borderline = get_memory_usage_mb()
        memory_dict['modules']['4_Borderline'] = {
            'before': initial_memory,
            'after': mem_borderline,
            'delta': mem_borderline - initial_memory
        }
        logger.info("     ✓ Borderline: F1 top-5=%s", ranked_by_f1_names_sensitivity[:5])
        
        monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR = monte_carlo_future.result()
        timing_dict['5_MonteCarlo'] = time.time() - overall_start
        mem_mc = get_memory_usage_mb()
        memory_dict['modules']['5_MonteCarlo'] = {
            'before': initial_memory,
            'after': mem_mc,
            'delta': mem_mc - initial_memory
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
        'delta': mem_after_agg - mem_before_agg
    }
    logger.info("  ✓ Aggregation: %.2fs", timing_dict['6_Aggregation'])
    
    # Track final memory and peak
    memory_dict['final'] = get_memory_usage_mb()
    memory_dict['peak'] = get_peak_memory_mb()
    logger.info(f"  💾 Final memory usage: {memory_dict['final']:.2f} MB (Peak: {memory_dict['peak']:.2f} MB)")

    # Persist results
    directory = f"myresults/robust_aggregated/{dataset}/{entity}/"
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
            'ga': {'f1': best_f1, 'pr_auc': best_pr_auc, 'fitness': best_fitness},
            'thompson': thompson_model_names,
            'gan': {
                'f1_names': Gan_ranked_by_f1_names, 
                'pr_auc_names': Gan_ranked_by_pr_auc_names,
                'f1_scores': Gan_ranked_by_f1,
                'pr_auc_scores': Gan_ranked_by_pr_auc,
                'best_model': Gan_ranked_by_f1_names[0] if len(Gan_ranked_by_f1_names) > 0 else 'N/A',
                'best_f1': Gan_ranked_by_f1[0][1][0]['f1'] if len(Gan_ranked_by_f1) > 0 else 0.0,
                'best_pr_auc': Gan_ranked_by_pr_auc[0][1][0]['pr_auc'] if len(Gan_ranked_by_pr_auc) > 0 else 0.0,
            },
            'borderline': {
                'f1_names': ranked_by_f1_names_sensitivity, 
                'pr_auc_names': ranked_by_pr_auc_names_sensitivity,
                'f1_scores': ranked_by_f1,
                'pr_auc_scores': ranked_by_pr_auc,
                'best_model': ranked_by_f1_names_sensitivity[0] if len(ranked_by_f1_names_sensitivity) > 0 else 'N/A',
                'best_f1': ranked_by_f1[0][1][0]['f1'] if len(ranked_by_f1) > 0 else 0.0,
                'best_pr_auc': ranked_by_pr_auc[0][1][0]['pr_auc'] if len(ranked_by_pr_auc) > 0 else 0.0,
            },
            'monte_carlo': {
                'f1_names': monte_carlo_ranked_models_F1, 
                'pr_auc_names': monte_carlo_ranked_models_PR,
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
    output_dir = Path(f"results/{dataset}/{entity}")
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

def find_num_falses(adjusted_y_pred_ind_current, test_data_copy, dataset, entity, values,
                    full_aggregated, best_ensemble, iteration):
    """
    Compute and persist the number of misclassifications for current single model(s) and ensemble.
    """
    misclassified_current = []
    for predicts in adjusted_y_pred_ind_current:
        true_values = np.array(test_data_copy.entities[0].labels)
        predicted_int = np.array(predicts).astype(int)
        incorrect = predicted_int != true_values
        misclassified_current.append(int(np.sum(incorrect)))

    misclassified_ensemble = []
    for predicts in [values[3]]:
        true_values = np.array(values[4])
        predicted_int = np.array(predicts).astype(int)
        incorrect = predicted_int != true_values
        misclassified_ensemble.append(int(np.sum(incorrect)))

    directory = f"myresults/robust_aggregated/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(
        directory, f"new_robust_aggregated_results_{dataset}_{entity}_{iteration}.txt"
    )
    with open(output_file, 'w') as f:
        f.write("Summary of falses:\n")
        f.write(f"chosen model (aggregated): {full_aggregated}\n")
        f.write(f"misclassified_current: {misclassified_current}\n")
        f.write("Falses for the ensemble:\n")
        f.write(f"chosen ensemble: {best_ensemble}\n")
        f.write(f"misclassified_ensemble: {misclassified_ensemble}\n")

# ------------------------------------------------------------------------------
# Background Re-optimization
# ------------------------------------------------------------------------------

def perform_reoptimization_task(
    window_idx, cumulative_online_windows, offline_data, offline_targets, offline_mask,
    test_data, train_data, dataset, entity, trained_models, loaded_model_names,
    anomaly_list, individual_predictions, base_model_predictions_train, 
    base_model_predictions_test, y_true_train, y_true_test, meta_model_type,
    use_parallel, test_data_before, logger, current_best_ensemble, current_best_single,
    algorithm_list_instances
):
    """
    Background task for re-optimization.
    Returns: (candidate_best_ensemble, candidate_best_single, individual_predictions,
              base_model_predictions_train, base_model_predictions_test, 
              y_true_train, y_true_test, meta_model_type, reopt_time)
    """
    reopt_start = time.time()
    logger.info(f"  🔄 [BACKGROUND] Starting re-optimization for window {window_idx}...")
    
    # ========== SLIDING WINDOW RE-TRAINING LOGIC ==========
    # 1. Concatenate the accumulated online windows
    online_concat_data = np.concatenate([w['data'] for w in cumulative_online_windows], axis=1)
    online_concat_labels = np.concatenate([w['labels'] for w in cumulative_online_windows], axis=1) \
        if cumulative_online_windows[0]['labels'].ndim > 1 else \
        np.concatenate([w['labels'].flatten() for w in cumulative_online_windows])
    online_concat_mask = np.concatenate([w['mask'] for w in cumulative_online_windows], axis=1)
    online_samples_count = online_concat_data.shape[1]
    
    # 2. Drop same number of samples from BEGINNING of offline data (sliding window)
    adjusted_offline_data = offline_data[:, online_samples_count:]
    adjusted_offline_labels = offline_targets[:, online_samples_count:] if offline_targets.ndim > 1 else offline_targets[online_samples_count:]
    adjusted_offline_mask = offline_mask[:, online_samples_count:]
    
    # 3. Create new training/test data by combining adjusted offline + new online
    sliding_test_data = np.concatenate([adjusted_offline_data, online_concat_data], axis=1)
    sliding_test_labels = np.concatenate([adjusted_offline_labels.flatten(), online_concat_labels.flatten()]).reshape(1, -1) \
        if adjusted_offline_labels.ndim > 1 or online_concat_labels.ndim > 1 else \
        np.concatenate([adjusted_offline_labels, online_concat_labels])
    sliding_test_mask = np.concatenate([adjusted_offline_mask, online_concat_mask], axis=1)
    
    logger.info(f"  📏 [BACKGROUND] Sliding window: dropped {online_samples_count} old, added {online_samples_count} new samples")
    logger.info(f"  📏 [BACKGROUND] New test data size: {sliding_test_data.shape[1]} samples")
    
    # 4. Update test_data with sliding window
    test_data_sliding = copy.deepcopy(test_data)
    test_data_sliding.entities[0].Y = sliding_test_data
    test_data_sliding.entities[0].labels = sliding_test_labels
    test_data_sliding.entities[0].mask = sliding_test_mask
    test_data_sliding.entities[0].n_time = sliding_test_data.shape[1]
    test_data_sliding.total_time = sliding_test_data.shape[1]
    
    # 5. Use REAL DATA without injection for online phase
    logger.info(f"  ⚠️  [BACKGROUND] Using REAL data (no injection) for adaptation")
    test_data_new_sliding = copy.deepcopy(test_data_sliding)
    
    # 6. Re-run model selection with sliding window data (SKIP GAN for speed)
    if use_parallel:
        (best_thompson, robust_agg, full_aggregated, best_ensemble,
         individual_predictions_new, base_model_predictions_train_new, base_model_predictions_test_new,
         y_true_train_new, y_true_test_new, meta_model_type_new, _) = run_model_selection_algorithms_2(
            train_data, test_data_new_sliding, dataset, entity, iteration=window_idx,
            trained_models=trained_models, model_list=loaded_model_names,
            test_data_gan=test_data_before, skip_gan=True, explain=explain
        )
    else:
        (best_thompson, robust_agg, full_aggregated, best_ensemble,
         individual_predictions_new, base_model_predictions_train_new, base_model_predictions_test_new,
         y_true_train_new, y_true_test_new, meta_model_type_new, _) = run_model_selection_algorithms_1(
            train_data, test_data_new_sliding, dataset, entity, iteration=window_idx,
            model_list=loaded_model_names, test_data_gan=test_data_before, skip_gan=True, explain=explain
        )
    
    candidate_best_ensemble = best_ensemble.copy()
    candidate_best_single = full_aggregated[0] if isinstance(full_aggregated, (list, tuple)) else full_aggregated
    
    reopt_time = time.time() - reopt_start
    logger.info(f"  ✓ [BACKGROUND] Re-optimization completed in {reopt_time:.2f}s")
    
    return (candidate_best_ensemble, candidate_best_single, individual_predictions_new,
            base_model_predictions_train_new, base_model_predictions_test_new,
            y_true_train_new, y_true_test_new, meta_model_type_new, reopt_time)

# ------------------------------------------------------------------------------
# Main Runner
# ------------------------------------------------------------------------------

def run_app(algorithm_list, algorithm_list_instances):
    args = get_args_from_cmdline()

    # Get dataset and entity from args (command line overrides, or use defaults)
    dataset = args.get('dataset', 'skab')
    entity = str(args.get('entity', '3'))
    use_parallel = args.get('parallel', False)
    enable_online_phase = args.get('enable_online', False)  # Online phase OFF by default
    iteration = args.get('iteration', 5)  # Default to 5 iterations for window sizing
    strategy = args.get('strategy', 'adaptive')  # adaptive, fixed-best, or fixed-random
    inject_online_regime = args.get('inject_online_regime', False)  # Regime shifts on online data only
    max_online_windows = args.get('max_online_windows', None)  # Limit online windows (None = no limit)
    explain = args.get('explain', False)  # Explainability OFF by default; --explain enables it
    stages = set(args.get('stages', ALL_STAGES))  # Which stage-6 sub-stages to run
    is_partial = stages != ALL_STAGES  # Strict subset → partial run (stop after selected stages)

    data_dir = args['dataset_path']
    
    logger.info("="*80)
    logger.info(f"🚀 STARTING RAMSeS EXECUTION: dataset={dataset}, entity={entity}, parallel={use_parallel}, online_phase={enable_online_phase}, iteration={iteration}, strategy={strategy}, online_regime={inject_online_regime}, max_windows={max_online_windows}")
    logger.info("="*80)
    
    logger.info("📂 STAGE 1/7: Loading Training Data...")
    train_data = load_data(
        dataset=dataset, group='train',
        entities=entity, downsampling=10,
        min_length=256, root_dir=data_dir, normalize=True, verbose=False
    )
    logger.info(f"✓ Training data loaded: {len(train_data.entities)} entity(ies)")
    
    logger.info("📂 STAGE 2/7: Loading Test Data...")
    test_data = load_data(
        dataset=dataset, group='test',
        entities=entity, downsampling=10,
        min_length=256, root_dir=data_dir, normalize=True, verbose=False
    )
    logger.info(f"✓ Test data loaded: {len(test_data.entities)} entity(ies)")

    if not train_data.entities:
        logger.error("Failed to load training data. Check dataset and paths.")
        return
    if not test_data.entities:
        logger.error("Failed to load test data. Check dataset and paths.")
        return

    logger.info("🔧 STAGE 3/7: Training/Loading Models...")
    model_trainer = TrainModels(
        dataset=dataset,
        entity=entity,
        algorithm_list=algorithm_list,
        downsampling=args['downsampling'],
        min_length=args['min_length'],
        root_dir=args['dataset_path'],
        training_size=args['training_size'],
        overwrite=args['overwrite'],
        verbose=args['verbose'],
        save_dir=args['trained_model_path'],
    )

    try:
        # Train (no-op if already present, depending on TrainModels implementation)
        logger.info("  → Checking existing models and training missing ones...")
        model_trainer.train_models(model_architectures=args['model_architectures'])
        logger.info("  ✓ Model training phase complete")

        # Load trained models using the dynamic path (base_dir/dataset/entity/)
        logger.info("  → Loading trained models from disk...")
        global trained_models
        models_dir = os.path.join(args['trained_model_path'], dataset, entity)
        trained_models = load_trained_models(algorithm_list_instances, models_dir)
        if not trained_models:
            raise ValueError("No models loaded. Check model paths and ensure models are trained.")
        
        # Filter algorithm_list_instances to only include successfully loaded models
        loaded_model_names = list(trained_models.keys())
        logger.info(f"✓ Loaded {len(loaded_model_names)} models: {', '.join(loaded_model_names)}")

        logger.info("💉 STAGE 4/7: Injecting Synthetic Anomalies...")
        # Inject anomalies
        anomaly_list = ['spikes']
        test_data_before = copy.deepcopy(test_data)
        train_data_before = copy.deepcopy(train_data)

        train_data, _ = Inject(train_data, anomaly_list)
        test_data, anomaly_sizes = Inject(test_data, anomaly_list)
        logger.info(f"✓ Injected anomalies: {anomaly_list}")

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

        out_dir = f"myresults/GA_Ens/{dataset}/{entity}/"
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"ensemble_scores_{dataset}_{entity}_Data_vs_anomalies_{anomaly_list}.png")
        plt.savefig(out_file, dpi=300)
        logger.info(f"✓ Visualization saved to {out_file}")

        logger.info("🔍 STAGE 6/7: Running Model Selection Algorithms...")
        logger.info("  This includes: GA, Thompson Sampling, GAN, Monte Carlo, Off-by-Threshold")
        
        # Data splitting: 80% for offline phase, 20% for online phase (as per paper)
        data = test_data_before.entities[0].Y
        targets = test_data_before.entities[0].labels
        mask = test_data_before.entities[0].mask
        total_samples = int(np.size(targets.flatten()))
        
        # Split: 80% offline (for model selection), 20% online (for streaming evaluation)
        offline_size = int(total_samples * 0.8)
        online_size = total_samples - offline_size
        
        # For offline phase: use first 80% of data
        offline_data = data[:, :offline_size]
        offline_targets = targets.flatten()[:offline_size].reshape(1, -1) if targets.ndim > 1 else targets[:offline_size]
        offline_mask = mask[:, :offline_size]
        
        # For online phase: sliding windows over remaining 20%
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
            # Inject regime shifts (scale + wander) 
            online_dataset, regime_sizes = Inject(online_dataset, ['scale', 'wander'])
            # Extract modified data
            online_data = online_dataset.entities[0].Y
            online_targets = online_dataset.entities[0].labels
            logger.info(f"  ✓ Regime shifts injected: scale + wander (sizes: {regime_sizes})")
        
        # Online phase: create sliding windows
        # Window size is determined by number of iterations from the 20% online data
        # iteration parameter determines how many windows we want: window_size = online_size / iteration
        # step_size = 5% of window size for 95% overlap between consecutive windows
        window_size = max(int(online_size / iteration), 100) if iteration > 0 else max(int(online_size * 0.2), 100)
        step_size = max(int(window_size * 0.05), 5)  # 5% of window size, at least 5
        
        logger.info(f"  📊 Data split: Total={total_samples}, Offline={offline_size} (80%), Online={online_size} (20%)")
        logger.info(f"  📊 Online windows: window_size={window_size} (online_size/{iteration}), step_size={step_size} (5% of window)")
        
        data_windows, targets_windows, new_mask, num_windows = initialize_sliding_windows(
            online_data, online_targets, online_mask,
            window_size, step_size
        )
        
        logger.info(f"  📊 Created {num_windows} sliding windows for online phase")
        
        # ==================== OFFLINE PHASE ====================
        logger.info("  🔧 OFFLINE PHASE: Running full model selection on 80% of data...")
        
        # Use offline data for initial model selection
        test_data.entities[0].Y = offline_data
        test_data.entities[0].labels = offline_targets
        test_data.entities[0].mask = offline_mask
        test_data.entities[0].n_time = offline_size
        test_data.total_time = offline_size

        test_data_new = copy.deepcopy(test_data)
        logger.info(f"  📏 Before Inject: test_data size = {test_data.entities[0].labels.shape[1] if test_data.entities[0].labels.ndim > 1 else test_data.entities[0].labels.shape[0]}")
        test_data_new, _ = Inject(test_data_new, anomaly_list)
        
        # Log data sizes for architecture verification
        test_data_before_size = test_data_before.entities[0].labels.shape[1] if test_data_before.entities[0].labels.ndim > 1 else test_data_before.entities[0].labels.shape[0]
        test_data_new_size = test_data_new.entities[0].labels.shape[1] if test_data_new.entities[0].labels.ndim > 1 else test_data_new.entities[0].labels.shape[0]
        logger.info(f"  📏 After Inject: test_data_before (full, for GAN) = {test_data_before_size}, test_data_new (injected, for others) = {test_data_new_size}")

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
                test_data_gan=test_data_before, explain=explain
            )
        else:
            mode = f"PARTIAL: {','.join(sorted(stages))}" if is_partial else "SEQUENTIAL mode"
            logger.info(f"  ⏱ Starting model selection pipeline ({mode})...")
            (best_thompson, robust_agg, full_aggregated, best_ensemble,
             individual_predictions, base_model_predictions_train, base_model_predictions_test,
             y_true_train, y_true_test, meta_model_type, extra_results) = run_model_selection_algorithms_1(
                train_data, test_data_new, dataset, entity, iteration=OFFLINE_ITERATION,
                model_list=loaded_model_names, test_data_gan=test_data_before, explain=explain,
                stages=stages
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
        _, single_model_scores, single_f1_list, single_pr_list = evaluate_individual_models(
            [best_single_model], test_data_for_eval, trained_models
        )
        single_model_f1 = to_scalar(single_f1_list[0]) if len(single_f1_list) > 0 else 0.0
        single_model_pr_auc = to_scalar(single_pr_list[0]) if len(single_pr_list) > 0 else 0.0
        
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
        
        # Framework decision: Choose ensemble if its F1 >= single model F1
        framework_choice = 'ensemble' if ensemble_f1 >= single_model_f1 else 'single_model'
        
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
                'ensemble_f1': ensemble_f1,
                'ensemble_pr_auc': ensemble_pr_auc,
                'ensemble_fitness': extra_results['ga']['fitness'],
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
                }
                # The narration ran after the algorithms' final/peak snapshot;
                # refresh so the report reflects the whole pipeline.
                memory_dict['final'] = get_memory_usage_mb()
                memory_dict['peak'] = get_peak_memory_mb()
            e2e_time = time.time() - e2e_start_time
            timing_dict['total'] = e2e_time

        logger.info("📝 STAGE 7/7: Writing Comprehensive Results...")
        # Write comprehensive results
        comp_results_dir = f"myresults/comprehensive/{dataset}/{entity}/"
        os.makedirs(comp_results_dir, exist_ok=True)
        comp_results_file = os.path.join(
            comp_results_dir, f"comprehensive_results_{dataset}_{entity}_iter{iteration}.txt"
        )
        write_comprehensive_results(comp_results_file, dataset, entity, iteration, results_dict, timing_dict, memory_dict)
        logger.info(f"✓ Results written to: {comp_results_file}")
        logger.info("="*80)
        logger.info(f"🎉 EXECUTION COMPLETE! Total Time: {e2e_time:.4f}s ({e2e_time/60:.2f} min)")
        logger.info("="*80)

        # ==================== ONLINE PHASE ====================
        # Real-time evaluation loop on remaining 20% of data using sliding windows
        # update_interval controls how often we re-run the expensive model selection pipeline
        enable_online_phase = args.get('enable_online', False)  # Off by default
        update_interval = args.get('update_interval', 5)  # Re-optimize every N windows (default: 5)
        
        # Save initial offline phase selection (window 0)
        save_current_selection(
            dataset, entity, window_idx=0, best_ensemble=best_ensemble,
            best_single_model=full_aggregated[0] if isinstance(full_aggregated, (list, tuple)) else full_aggregated,
            meta_model_type=meta_model_type, trained_models=trained_models,
            individual_predictions=individual_predictions,
            base_model_predictions_train=base_model_predictions_train,
            base_model_predictions_test=base_model_predictions_test,
            y_true_train=y_true_train, y_true_test=y_true_test
        )
        
        # ========== STRATEGY SETUP ==========
        # For fixed strategies, lock in the models now (no re-optimization during online phase)
        if strategy == 'fixed-best':
            logger.info(f"  🔒 STRATEGY: FIXED-BEST - Using best offline models throughout (no re-optimization)")
            fixed_ensemble = best_ensemble.copy()
            fixed_single_model = full_aggregated[0] if isinstance(full_aggregated, (list, tuple)) else full_aggregated
        elif strategy == 'fixed-random':
            logger.info(f"  🎲 STRATEGY: FIXED-RANDOM - Randomly selecting fixed model (no re-optimization)")
            import random
            available_models = list(trained_models.keys())
            fixed_single_model = random.choice(available_models)
            # Random ensemble: pick 3-6 random models
            ensemble_size = random.randint(3, min(6, len(available_models)))
            fixed_ensemble = random.sample(available_models, ensemble_size)
            logger.info(f"  🎲 Fixed random model: {fixed_single_model}")
            logger.info(f"  🎲 Fixed random ensemble: {fixed_ensemble}")
        else:  # adaptive
            logger.info(f"  🔄 STRATEGY: ADAPTIVE - Will re-optimize every {update_interval} windows")
        
        if enable_online_phase:
            logger.info(f"📊 ONLINE PHASE ENABLED: {num_windows} windows, update_interval={update_interval}")
            from tqdm import tqdm
        else:
            logger.info(f"  ℹ️  Online phase disabled (20% of data reserved but not processed)")
            logger.info(f"  ℹ️  To enable online phase, add --enable_online flag")
        
        if enable_online_phase and num_windows > 1:
            online_start_time = time.time()
            i = 1
            # Track cumulative online windows for sliding window re-training
            cumulative_online_windows = []
            
            # Apply max_online_windows limit if specified
            effective_num_windows = num_windows if max_online_windows is None else min(num_windows, max_online_windows + 1)
            if max_online_windows is not None:
                logger.info(f"⚠️  LIMITING online phase to {max_online_windows} windows (out of {num_windows-1} available)")
            
            # Create progress bar for online windows
            pbar = tqdm(total=effective_num_windows-1, desc="Online Windows", unit="window", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            # Initialize background re-optimization executor and tracking
            reopt_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            pending_reopt_future = None  # Track the running background task
            pending_reopt_window = None  # Track which window triggered the reopt
            
            while i < effective_num_windows:
                pbar.set_description(f"Window {i}/{effective_num_windows-1}")
                logger.info(f"🔄 Processing online window {i}/{effective_num_windows-1}")
                current_window_data = data_windows[i]
                current_window_labels = targets_windows[i]
                current_window_mask = new_mask[i]
                
                # Accumulate this window for potential re-training
                cumulative_online_windows.append({
                    'data': current_window_data,
                    'labels': current_window_labels,
                    'mask': current_window_mask
                })
                
                # Set up test data for current window evaluation
                test_data.entities[0].Y = current_window_data
                test_data.entities[0].labels = current_window_labels
                test_data.entities[0].mask = current_window_mask
                test_data.entities[0].n_time = int(np.size(current_window_labels.flatten()))
                test_data.total_time = int(np.size(current_window_labels.flatten()))

                # ========== CHECK FOR COMPLETED BACKGROUND RE-OPTIMIZATION ==========
                # If we have a pending re-optimization task, check if it's done
                if pending_reopt_future is not None and pending_reopt_future.done():
                    try:
                        # Get results from completed background task
                        (candidate_best_ensemble, candidate_best_single, individual_predictions_new,
                         base_model_predictions_train_new, base_model_predictions_test_new,
                         y_true_train_new, y_true_test_new, meta_model_type_new, reopt_time) = pending_reopt_future.result()
                        
                        logger.info(f"  ✅ [BACKGROUND] Re-optimization completed for window {pending_reopt_window} in {reopt_time:.2f}s")
                        logger.info(f"  📊 Applying results from background re-optimization...")
                        
                        # Apply the new models (they've already been selected as best)
                        best_ensemble = candidate_best_ensemble
                        if isinstance(full_aggregated, (list, tuple)):
                            full_aggregated = (candidate_best_single,) + tuple(full_aggregated[1:])
                        else:
                            full_aggregated = candidate_best_single
                        
                        # Update predictions and meta-model
                        individual_predictions = individual_predictions_new
                        base_model_predictions_train = base_model_predictions_train_new
                        base_model_predictions_test = base_model_predictions_test_new
                        y_true_train = y_true_train_new
                        y_true_test = y_true_test_new
                        meta_model_type = meta_model_type_new
                        
                        logger.info(f"  → Updated ensemble: {best_ensemble}")
                        logger.info(f"  → Updated single model: {candidate_best_single}")
                        
                        # Save the updated selection
                        save_current_selection(
                            dataset, entity, window_idx=pending_reopt_window, best_ensemble=best_ensemble,
                            best_single_model=candidate_best_single,
                            meta_model_type=meta_model_type, trained_models=trained_models,
                            individual_predictions=individual_predictions,
                            base_model_predictions_train=base_model_predictions_train,
                            base_model_predictions_test=base_model_predictions_test,
                            y_true_train=y_true_train, y_true_test=y_true_test
                        )
                        
                    except Exception as e:
                        logger.error(f"  ❌ Background re-optimization failed: {e}")
                        logger.error(f"  ↩️  Keeping current models")
                    finally:
                        # Clear the pending task
                        pending_reopt_future = None
                        pending_reopt_window = None
                
                # ========== TRIGGER NEW BACKGROUND RE-OPTIMIZATION IF NEEDED ==========
                # EXPENSIVE: Re-run full model selection pipeline only every N windows
                # Strategy check: only re-optimize if strategy is 'adaptive'
                if strategy == 'adaptive' and i % update_interval == 0 and pending_reopt_future is None:
                    logger.info(f"  🔄 Triggering BACKGROUND re-optimization at window {i} (every {update_interval} windows)...")
                    
                    # ========== EVALUATE CURRENT MODELS (FOR LOGGING) ==========
                    logger.info(f"  📊 Evaluating CURRENT models on window {i}...")
                    
                    # Save current models
                    current_best_ensemble = best_ensemble.copy()
                    current_best_single = full_aggregated[0] if isinstance(full_aggregated, (list, tuple)) else full_aggregated
                    
                    logger.info(f"  → Current single model: {current_best_single}")
                    logger.info(f"  → Current ensemble: {current_best_ensemble}")
                    logger.info(f"  → Starting re-optimization in background (will apply when ready)...")
                    
                    # ========== SUBMIT BACKGROUND RE-OPTIMIZATION TASK ==========
                    pending_reopt_future = reopt_executor.submit(
                        perform_reoptimization_task,
                        i, cumulative_online_windows, offline_data, offline_targets, offline_mask,
                        test_data, train_data, dataset, entity, trained_models, loaded_model_names,
                        anomaly_list, individual_predictions, base_model_predictions_train,
                        base_model_predictions_test, y_true_train, y_true_test, meta_model_type,
                        use_parallel, test_data_before, logger, current_best_ensemble, current_best_single,
                        algorithm_list_instances
                    )
                    pending_reopt_window = i
                    logger.info(f"  🚀 Background re-optimization task submitted for window {i}")
                    
                    # Continue immediately with evaluation using current models (don't wait)
                    did_reoptimization = False  # We'll evaluate normally since reopt is async
                else:
                    # No re-optimization: either because not at update_interval or strategy is fixed
                    if strategy in ['fixed-best', 'fixed-random']:
                        # Use the fixed models (set at beginning)
                        best_ensemble = fixed_ensemble
                        if isinstance(full_aggregated, (list, tuple)):
                            full_aggregated = (fixed_single_model,) + tuple(full_aggregated[1:])
                        else:
                            full_aggregated = fixed_single_model
                        logger.info(f"  🔒 Using fixed models (strategy={strategy})")
                    else:
                        # Adaptive but not at update interval yet
                        logger.info(f"  ⏩ Skipping re-optimization (next at window {((i // update_interval) + 1) * update_interval})")
                    did_reoptimization = False
                
                # Only evaluate if we didn't just do re-optimization
                # (At reoptimization windows, evaluation already happened in Step 3)
                if not did_reoptimization:
                    logger.info(f"  📊 Evaluating models on window {i}...")
                    # Restrict to best ensemble's models
                    trained_models_new = {}
                    algorithm_list_new = []
                    for model in best_ensemble:
                        trained_models_new[model] = trained_models[model]
                        algorithm_list_new.append(model)

                    test_data_new = copy.deepcopy(test_data)
                    test_data_new, _ = Inject(test_data_new, anomaly_list)

                    # LIGHTWEIGHT: Evaluate current best single model on this window
                    test_data_new_copy = copy.deepcopy(test_data_new)
                    _, adjusted_y_pred_ind_current, _, _ = evaluate_individual_models(
                        [full_aggregated[0]], test_data_new_copy, trained_models
                    )

                    # LIGHTWEIGHT: Evaluate ensemble fitness on current window
                    test_data_new_copy = copy.deepcopy(test_data_new)
                    values = fitness_function(
                        best_ensemble, train_data, test_data_new_copy, trained_models_new,
                        individual_predictions, base_model_predictions_train, algorithm_list_instances,
                        base_model_predictions_test, y_true_train, y_true_test,
                        meta_model_type=meta_model_type
                    )

                    # Persist falses (TP/FP/FN tracking)
                    test_data_new_copy = copy.deepcopy(test_data_new)
                    find_num_falses(
                        adjusted_y_pred_ind_current, test_data_new_copy, dataset, entity, values,
                        full_aggregated[0], best_ensemble, iteration=i
                    )
                
                i += 1
                pbar.update(1)
            
            pbar.close()
            
            # ========== WAIT FOR ANY PENDING BACKGROUND RE-OPTIMIZATION ==========
            if pending_reopt_future is not None:
                logger.info(f"  ⏳ Waiting for pending background re-optimization to complete...")
                try:
                    (candidate_best_ensemble, candidate_best_single, individual_predictions_new,
                     base_model_predictions_train_new, base_model_predictions_test_new,
                     y_true_train_new, y_true_test_new, meta_model_type_new, reopt_time) = pending_reopt_future.result()
                    
                    logger.info(f"  ✅ Final background re-optimization completed in {reopt_time:.2f}s")
                    # Apply the results
                    best_ensemble = candidate_best_ensemble
                    if isinstance(full_aggregated, (list, tuple)):
                        full_aggregated = (candidate_best_single,) + tuple(full_aggregated[1:])
                    else:
                        full_aggregated = candidate_best_single
                    individual_predictions = individual_predictions_new
                    base_model_predictions_train = base_model_predictions_train_new
                    base_model_predictions_test = base_model_predictions_test_new
                    y_true_train = y_true_train_new
                    y_true_test = y_true_test_new
                    meta_model_type = meta_model_type_new
                    
                    logger.info(f"  → Final ensemble: {best_ensemble}")
                    logger.info(f"  → Final single model: {candidate_best_single}")
                except Exception as e:
                    logger.error(f"  ❌ Final background re-optimization failed: {e}")
            
            # Shutdown the executor
            reopt_executor.shutdown(wait=True)
            logger.info("  ✓ Background re-optimization executor shut down")
            
            online_end_time = time.time()
            online_total_time = online_end_time - online_start_time
            
            logger.info("✓ Online phase completed")
            logger.info(f"⏱️  Online Phase Total Time: {online_total_time:.2f}s ({online_total_time/60:.2f} min)")
            
            # Write online phase summary
            online_summary_file = f"myresults/comprehensive/{dataset}/{entity}/online_phase_timing.txt"
            with open(online_summary_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("RAMSeS Online Phase Timing Summary\n")
                f.write("="*80 + "\n")
                f.write(f"Dataset: {dataset}\n")
                f.write(f"Entity: {entity}\n")
                f.write(f"Iteration: {iteration}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                f.write(f"Total Windows Processed: {num_windows-1}\n")
                f.write(f"Update Interval: {update_interval}\n")
                f.write(f"Total Online Time: {online_total_time:.2f}s ({online_total_time/60:.2f} min)\n")
                f.write(f"Average Time per Window: {online_total_time/(num_windows-1):.2f}s\n")
                f.write(f"Offline Phase Time: {e2e_time:.2f}s ({e2e_time/60:.2f} min)\n")
                f.write(f"Grand Total Time: {online_total_time + e2e_time:.2f}s ({(online_total_time + e2e_time)/60:.2f} min)\n")
                f.write("="*80 + "\n")
            
            logger.info(f"📝 Online timing saved to: {online_summary_file}")
        else:
            logger.info("  ℹ️  Online phase skipped (insufficient windows)")

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
