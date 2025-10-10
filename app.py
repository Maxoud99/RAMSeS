# time_series_framework/app.py
import concurrent.futures
import copy
import logging
import os
import traceback

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
from Model_Selection.rank_aggregation import enhanced_markov_chain_rank_aggregator_text
from Model_Training.train import TrainModels
from Utils.utils import get_args_from_cmdline

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# save_dir = "Mononito/trained_models/anomaly_archive/031_UCR_Anomaly_DISTORTEDInternalBleeding20/"
save_dir = "/home/maxoud/projects/RAMS-TSAD/Mononito/trained_models/smd/machine-3-10/"

algorithm_list = ['DGHL', 'LSTMVAE', 'NN', 'RNN', 'MD', 'RM', 'LOF', 'CBLOF']
algorithm_list_instances = ['LOF_1', 'LOF_2', 'LOF_3', 'LOF_4', 'NN_1', 'NN_2', 'NN_3', 'RNN_1', 'RNN_2', 'RNN_3', 'RNN_4',
    'CBLOF_1', 'CBLOF_2', 'CBLOF_3', 'CBLOF_4',
    'DGHL_1', 'DGHL_2', 'DGHL_3', 'DGHL_4',
    'LOF_1', 'LOF_2', 'LOF_3', 'LOF_4',
    'LSTMVAE_1', 'LSTMVAE_2', 'LSTMVAE_3', 'LSTMVAE_4',
    'MD_1',
    'RM_1', 'RM_2', 'RM_3'
]

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

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
    for name in model_names:
        path = os.path.join(models_dir, f"{name}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model {name} not found in {models_dir}")
        with open(path, 'rb') as fh:
            model = t.load(fh, weights_only=False)
            try:
                model.eval()
            except AttributeError:
                # If the checkpoint isn't a nn.Module, we still store it as-is.
                pass
            trained[name] = model
    logger.info("Loaded %d trained models from %s", len(trained), models_dir)
    return trained

# ------------------------------------------------------------------------------
# Model-Selection Pipelines
# ------------------------------------------------------------------------------

def run_model_selection_algorithms_1(train_data, test_data, dataset, entity, iteration):
    """
    One-pass model selection pipeline in the order:
      1) GA (stacking ensemble search)
      2) Thompson Sampling (LinTS)
      3) GAN robustness test
      4) Off-by-threshold (borderline sensitivity)
      5) Monte Carlo (noise stress test)
      6) Rank aggregations (robust-only, then merged with Thompson)

    Returns (exactly 10 items, preserved):
        thompson_model_names[0],
        robust_agg[1],
        full_aggregated[1],
        best_ensemble,
        individual_predictions,
        base_model_predictions_train,
        base_model_predictions_test,
        y_true_train,
        y_true_test,
        meta_model_type
    """
    # -------------------------
    # 1) Genetic Algorithm (GA)
    # -------------------------
    best_ensemble, best_f1, best_pr_auc, best_fitness, \
    individual_predictions, base_model_predictions_train, base_model_predictions_test, \
    y_true_train, y_true_test, meta_model_type = genetic_algorithm(
        dataset, entity, train_data, test_data,
        algorithm_list_instances, trained_models,
        population_size=1, generations=1,
        meta_model_type='rf', mutation_rate=1,
    )
    logger.info(
        "[GA] Best ensemble=%s | F1=%.4f | PR-AUC=%.4f | fitness=%.4f",
        best_ensemble, best_f1, best_pr_auc, best_fitness
    )

    # -----------------------------------
    # 2) Thompson Sampling (LinTS, online)
    # -----------------------------------
    thompson_model_names = run_linear_thompson_sampling(
        test_data=test_data,
        trained_models=trained_models,
        model_names=algorithm_list_instances,
        dataset=dataset,
        entity=entity,
        iterations=50,
        iteration=iteration,
    )
    logger.info("[Thompson] Top-5: %s", thompson_model_names[:5])

    # -------------------------
    # 3) GAN Robustness Testing
    # -------------------------
    gan_results = run_Gan(
        test_data, trained_models, algorithm_list_instances, dataset, entity
    )
    Gan_ranked_by_f1, Gan_ranked_by_pr_auc, \
    Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names = (
        gan_results[0], gan_results[1], gan_results[2], gan_results[3]
    )
    logger.info("[GAN] F1 names top-5: %s", Gan_ranked_by_f1_names[:5])
    logger.info("[GAN] PR names top-5: %s", Gan_ranked_by_pr_auc_names[:5])

    # --------------------------------------------
    # 4) Off-by-threshold (borderline sensitivity)
    # --------------------------------------------
    ranked_by_f1, ranked_by_pr_auc, \
    ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity = run_off_by_threshold(
        test_data, trained_models, algorithm_list_instances, dataset, entity
    )
    logger.info("[Borderline] F1 names top-5: %s", ranked_by_f1_names_sensitivity[:5])
    logger.info("[Borderline] PR names top-5: %s", ranked_by_pr_auc_names_sensitivity[:5])

    # ---------------------------------
    # 5) Monte Carlo (noise stress test)
    # ---------------------------------
    monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR = run_monte_carlo_simulation(
        test_data, trained_models, algorithm_list_instances, dataset, entity,
        n_simulations=2, noise_level=0.1,
    )
    logger.info("[MonteCarlo] F1 names top-5: %s", monte_carlo_ranked_models_F1[:5])
    logger.info("[MonteCarlo] PR names top-5: %s", monte_carlo_ranked_models_PR[:5])

    # -----------------------
    # 6) Rank Aggregations
    # -----------------------
    # Robust-only aggregation in the requested order: GAN → Borderline → Monte Carlo
    test_for_rank = [
        Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names,
        ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity,
        monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR,
    ]
    robust_agg = enhanced_markov_chain_rank_aggregator_text(test_for_rank)

    # Final merge of robust aggregation vs Thompson Sampling
    full_ = [robust_agg[1], thompson_model_names]
    full_aggregated = enhanced_markov_chain_rank_aggregator_text(full_)

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

    # Keep the 10-item tuple exactly as before
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
    )


def run_model_selection_algorithms_2(train_data, test_data, dataset, entity, iteration):
    """
    Parallel variant: executes GAN, Borderline, Monte Carlo, GA, and Thompson in parallel
    (not used by default in run_app).
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        monte_carlo_future = executor.submit(
            run_monte_carlo_simulation,
            test_data, trained_models, algorithm_list_instances,
            dataset, entity, 2, 0.1
        )
        off_by_threshold_future = executor.submit(
            run_off_by_threshold,
            test_data, trained_models, algorithm_list_instances,
            dataset, entity
        )
        gan_future = executor.submit(
            run_Gan,
            test_data, trained_models, algorithm_list_instances,
            dataset, entity
        )
        genetic_future = executor.submit(
            genetic_algorithm,
            dataset, entity, train_data, test_data,
            algorithm_list_instances, trained_models,
            5, 10, 'lr', 0.1
        )
        thompson_future = executor.submit(
            run_linear_thompson_sampling,
            test_data=test_data,
            trained_models=trained_models,
            model_names=algorithm_list_instances,
            dataset=dataset,
            entity=entity,
            iterations=50,
            iteration=iteration,
        )

        monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR = monte_carlo_future.result()
        ranked_by_f1, ranked_by_pr_auc, ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity = \
            off_by_threshold_future.result()

        Gan_ranked_by_f1, Gan_ranked_by_pr_auc, \
        Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names = gan_future.result()

        (best_ensemble, best_f1, best_pr_auc, best_fitness,
         individual_predictions, base_model_predictions_train, base_model_predictions_test,
         y_true_train, y_true_test, meta_model_type) = genetic_future.result()

        thompson_model_names = thompson_future.result()

    logger.info("[GA] Best ensemble=%s | F1=%.4f | PR-AUC=%.4f | fitness=%.4f",
                best_ensemble, best_f1, best_pr_auc, best_fitness)

    test_for_rank = [
        monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR,
        Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names,
        ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity,
    ]
    robust_agg = enhanced_markov_chain_rank_aggregator_text(test_for_rank)
    full_ = [robust_agg[1], thompson_model_names]
    full_aggregated = enhanced_markov_chain_rank_aggregator_text(full_)

    directory = f"myresults/robust_aggregated/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(
        directory, f"robust_aggregated_results_{dataset}_{entity}_{iteration}.txt"
    )
    with open(output_file, 'w') as f:
        f.write("Summary of robust tests:\n")
        f.write("\n[Monte Carlo]\n")
        f.write(f"{monte_carlo_ranked_models_F1}\n{monte_carlo_ranked_models_PR}\n")
        f.write("\n[GAN]\n")
        f.write(f"{Gan_ranked_by_f1_names}\n{Gan_ranked_by_pr_auc_names}\n")
        f.write("\n[Borderline]\n")
        f.write(f"{ranked_by_f1_names_sensitivity}\n{ranked_by_pr_auc_names_sensitivity}\n")
        f.write("\n[Robust rank aggregate]\n")
        f.write(f"{robust_agg}\n")
        f.write("\n[Final aggregate vs Thompson]\n")
        f.write(f"{full_aggregated}\n")

    return (
        thompson_model_names[0], robust_agg[1], full_aggregated[1],
        best_ensemble, individual_predictions, base_model_predictions_train,
        base_model_predictions_test, y_true_train, y_true_test, meta_model_type
    )

# ------------------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------------------

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
# Main Runner
# ------------------------------------------------------------------------------

def run_app(algorithm_list, algorithm_list_instances):
    args = get_args_from_cmdline()

    data_dir = args['dataset_path']
    train_data = load_data(
        dataset='smd', group='train',
        entities='machine-3-10', downsampling=10,
        min_length=256, root_dir=data_dir, normalize=True, verbose=False
    )
    test_data = load_data(
        dataset='smd', group='test',
        entities='machine-3-10', downsampling=10,
        min_length=256, root_dir=data_dir, normalize=True, verbose=False
    )

    if not train_data.entities:
        logger.error("Failed to load training data. Check dataset and paths.")
        return
    if not test_data.entities:
        logger.error("Failed to load test data. Check dataset and paths.")
        return

    dataset = 'smd'
    entity = 'machine-3-10'

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
        model_trainer.train_models(model_architectures=args['model_architectures'])

        # Load trained models (uses the global save_dir defined at top)
        global trained_models
        trained_models = load_trained_models(algorithm_list_instances, save_dir)
        if not trained_models:
            raise ValueError("No models loaded. Check model paths and ensure models are trained.")

        # Inject anomalies
        anomaly_list = ['spikes']
        test_data_before = copy.deepcopy(test_data)
        train_data_before = copy.deepcopy(train_data)

        train_data, _ = Inject(train_data, anomaly_list)
        test_data, anomaly_sizes = Inject(test_data, anomaly_list)

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

        # Sliding windows setup
        data = test_data_before.entities[0].Y
        targets = test_data_before.entities[0].labels
        mask = test_data_before.entities[0].mask
        iterations = 1  # Real-time loop count

        data_windows, targets_windows, new_mask, num_windows = initialize_sliding_windows(
            data, targets, mask,
            int(np.size(targets.flatten()) / iterations),
            int(np.size(targets.flatten()) / iterations) - 5,
        )

        # First window
        test_data.entities[0].Y = data_windows[0]
        test_data.entities[0].labels = targets_windows[0]
        test_data.entities[0].mask = new_mask[0]
        test_data.entities[0].n_time = int(np.size(targets_windows[0].flatten()))
        test_data.total_time = int(np.size(targets_windows[0].flatten()))

        test_data_new = copy.deepcopy(test_data)
        test_data_new, _ = Inject(test_data_new, anomaly_list)

        (best_thompson, robust_agg, full_aggregated, best_ensemble,
         individual_predictions, base_model_predictions_train, base_model_predictions_test,
         y_true_train, y_true_test, meta_model_type) = run_model_selection_algorithms_1(
            train_data, test_data_new, dataset, entity, iteration=0
        )

        # Real-time evaluation loop (iterations=1 means it won't run)
        i = 1
        while i < iterations:
            test_data.entities[0].Y = data_windows[i]
            test_data.entities[0].labels = targets_windows[i]
            test_data.entities[0].mask = new_mask[i]
            test_data.entities[0].n_time = int(np.size(targets_windows[i].flatten()))
            test_data.total_time = int(np.size(targets_windows[i].flatten()))

            # Restrict to best ensemble's models
            trained_models_new = {}
            algorithm_list_new = []
            for model in best_ensemble:
                trained_models_new[model] = trained_models[model]
                algorithm_list_new.append(model)

            test_data_new = copy.deepcopy(test_data)
            test_data_new, _ = Inject(test_data_new, anomaly_list)

            # Evaluate current best single model
            test_data_new_copy = copy.deepcopy(test_data_new)
            _, adjusted_y_pred_ind_current, _, _ = evaluate_individual_models(
                [full_aggregated[0]], test_data_new_copy, trained_models
            )

            # Evaluate ensemble fitness on current window
            test_data_new_copy = copy.deepcopy(test_data_new)
            values = fitness_function(
                best_ensemble, train_data, test_data_new_copy, trained_models_new,
                individual_predictions, base_model_predictions_train, algorithm_list_instances,
                base_model_predictions_test, y_true_train, y_true_test,
                meta_model_type=meta_model_type
            )

            # Persist falses
            test_data_new_copy = copy.deepcopy(test_data_new)
            find_num_falses(
                adjusted_y_pred_ind_current, test_data_new_copy, dataset, entity, values,
                full_aggregated[0], best_ensemble, iteration=i
            )

            # Update selection for next iteration
            test_data_new_copy = copy.deepcopy(test_data_new)
            (best_thompson, robust_agg, full_aggregated, best_ensemble,
             individual_predictions, base_model_predictions_train, base_model_predictions_test,
             y_true_train, y_true_test, meta_model_type) = run_model_selection_algorithms_1(
                train_data, test_data_new_copy, dataset, entity, iteration=i
            )
            i += 1

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
