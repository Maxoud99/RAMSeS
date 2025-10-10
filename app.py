# time_series_framework/app.py
import concurrent.futures
import copy
import logging
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from loguru import logger

from Datasets.load import load_data
from Metrics.Ensemble_GA import genetic_algorithm, evaluate_individual_models, fitness_function
from Model_Selection.Sensitivity_robustness.GAN_test import run_Gan
from Model_Selection.Sensitivity_robustness.Monte_Carlo_Simulation import run_monte_carlo_simulation
from Model_Selection.Sensitivity_robustness.off_by_threshold_testing import run_off_by_threshold
from Model_Selection.Thompson_Sampling import run_linear_thompson_sampling, initialize_sliding_windows
from Model_Selection.inject_anomalies import Inject
from Model_Selection.rank_aggregation import enhanced_markov_chain_rank_aggregator_text
from Model_Training.train import TrainModels
from Utils.utils import get_args_from_cmdline

# save_dir = "Mononito/trained_models/anomaly_archive/031_UCR_Anomaly_DISTORTEDInternalBleeding20/"
save_dir = "/home/maxoud/projects/RAMS-TSAD/Mononito/trained_models/smd/machine-3-10/"
# algorithm_list = ['DGHL', 'LSTMVAE', 'MD', 'RM', 'LOF', 'CBLOFd']
# algorithm_list_instances = ['CBLOF_1', 'CBLOF_2', 'CBLOF_3', 'CBLOF_4', 'DGHL_1', 'DGHL_2', 'DGHL_3', 'DGHL_4', 'LOF_1',
#                             'LOF_2', 'LOF_3', 'LOF_4', 'LSTMVAE_1', 'LSTMVAE_2', 'LSTMVAE_3', 'LSTMVAE_4', 'MD_1',
#                             'RM_1', 'RM_2', 'RM_3']
algorithm_list = ['LOF', 'NN', 'RNN']
algorithm_list_instances = ['LOF_1', 'LOF_2', 'LOF_3', 'LOF_4', 'NN_1', 'NN_2', 'NN_3', 'RNN_1', 'RNN_2', 'RNN_3',
                            'RNN_4']

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trained_models(algorithm_list, save_dir):
    """Load trained models from the save directory."""
    trained_models = {}
    for model_name in algorithm_list:
        model_path = os.path.join(save_dir, f"{model_name}.pth")

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = t.load(f, weights_only=False)
                model.eval()  # Set model to evaluation mode
                trained_models[model_name] = model
        else:
            raise FileNotFoundError(f"Model {model_name} not found in {save_dir}")
    return trained_models

def run_model_selection_algorithms_1(train_data, test_data, dataset, entity, iteration):

    # # Run genetic algorithm for model selection
    best_ensemble, best_f1, best_pr_auc, best_fitness, individual_predictions, base_model_predictions_train, base_model_predictions_test, y_true_train, y_true_test, meta_model_type = genetic_algorithm(
        dataset, entity, train_data, test_data,
        algorithm_list_instances, trained_models,
        population_size=1, generations=1,
        meta_model_type='rf', mutation_rate=1)
    logger.info(
        f"Best ensemble: {best_ensemble} with F1 score {best_f1}, PR AUC {best_pr_auc}, and fitness {best_fitness}")
    logger.info(
        f"Best ensemble: {best_ensemble} with F1 score {best_f1}, PR AUC {best_pr_auc}, and fitness {best_fitness}")
    return
    monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR = run_monte_carlo_simulation(test_data,
                                                                                            trained_models,
                                                                                            algorithm_list_instances,
                                                                                            dataset, entity,
                                                                                            n_simulations=2,
                                                                                            noise_level=0.1)

    # # Off by threshold testing
    ranked_by_f1, ranked_by_pr_auc, ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity = run_off_by_threshold(
        test_data,
        trained_models,
        algorithm_list_instances,
        dataset,
        entity)

    # GAN testing
    values = run_Gan(test_data, trained_models, algorithm_list_instances, dataset, entity)
    Gan_ranked_by_f1, Gan_ranked_by_pr_auc, Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names = values[0], values[
        1], values[2], values[3]

    logger.info("1- Monte carlo")
    logger.info(monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR)
    logger.info("2- Gan")
    logger.info(Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names)
    logger.info("4- Off by threshold testing (sensitivity)")
    logger.info(ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity)


    #
    # ### Thompson Sampling
    thompson_model_names = run_linear_thompson_sampling(
        test_data=test_data,
        trained_models=trained_models,
        model_names=algorithm_list_instances,
        dataset=dataset,
        entity=entity,
        iterations=50,
        iteration=iteration
    )

    # # # Rank Aggregation
    test_for_rank = [monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR,
                     Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names,
                     ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity
                     ]

    robust_agg = enhanced_markov_chain_rank_aggregator_text(test_for_rank)
    full_ = [robust_agg[1], thompson_model_names]
    full_aggregated = enhanced_markov_chain_rank_aggregator_text(full_)

    directory = f'myresults/robust_aggregated/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(directory, f"robust_aggregated_results_{dataset}_{entity}_{iteration}.txt")

    with open(output_file, 'w') as f:
        f.write("Summary of robust tests:\n")
        f.write("\n1- Monte carlo\n")
        f.write(f"{monte_carlo_ranked_models_F1} \n {monte_carlo_ranked_models_PR} \n")
        f.write("\n Gan\n")
        f.write(f"{Gan_ranked_by_f1_names} \n {Gan_ranked_by_pr_auc_names} \n")
        f.write("\n Off by threshold testing (sensitivity)\n")
        f.write(f"{ranked_by_f1_names_sensitivity} \n {ranked_by_pr_auc_names_sensitivity} \n")
        f.write(f"Robust rank aggregates\n")
        f.write(f"{robust_agg}\n")
        f.write(f"All aggregated\n")
        f.write(f"{full_aggregated}\n")

    return thompson_model_names[0], robust_agg[1], full_aggregated[
        1], best_ensemble, individual_predictions, base_model_predictions_train, base_model_predictions_test, y_true_train, y_true_test, meta_model_type


#
#
def run_model_selection_algorithms_2(train_data, test_data, dataset, entity, iteration):
    # Create a ThreadPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks for parallel execution

        # Monte Carlo Simulation
        monte_carlo_future = executor.submit(
            run_monte_carlo_simulation,
            test_data, trained_models, algorithm_list_instances,
            dataset, entity, n_simulations=2, noise_level=0.1
        )

        # Off by threshold testing
        off_by_threshold_future = executor.submit(
            run_off_by_threshold,
            test_data, trained_models, algorithm_list_instances,
            dataset, entity
        )

        # GAN testing
        gan_future = executor.submit(
            run_Gan,
            test_data, trained_models, algorithm_list_instances,
            dataset, entity
        )

        # Genetic algorithm
        genetic_future = executor.submit(
            genetic_algorithm,
            dataset, entity, train_data, test_data,
            algorithm_list_instances, trained_models,
            population_size=5, generations=10,
            meta_model_type='lr', mutation_rate=0.1
        )

        # Thompson Sampling
        thompson_future = executor.submit(
            run_linear_thompson_sampling,
            test_data=test_data,
            trained_models=trained_models,
            model_names=algorithm_list_instances,
            dataset=dataset,
            entity=entity,
            iterations=50,
            iteration=iteration
        )

        # Get results as they complete
        monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR = monte_carlo_future.result()

        ranked_by_f1, ranked_by_pr_auc, ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity = off_by_threshold_future.result()

        values = gan_future.result()
        Gan_ranked_by_f1, Gan_ranked_by_pr_auc, Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names = values[0], values[
            1], values[2], values[3]

        best_ensemble, best_f1, best_pr_auc, best_fitness, individual_predictions, base_model_predictions_train, base_model_predictions_test, y_true_train, y_true_test, meta_model_type = genetic_future.result()

        thompson_model_names = thompson_future.result()

    # Log results (after all parallel operations are complete)
    logger.info("1- Monte carlo")
    logger.info(monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR)
    logger.info("2- Gan")
    logger.info(Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names)
    logger.info("4- Off by threshold testing (sensitivity)")
    logger.info(ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity)
    logger.info(
        f"Best ensemble: {best_ensemble} with F1 score {best_f1}, PR AUC {best_pr_auc}, and fitness {best_fitness}")
    logger.info(
        f"Best ensemble: {best_ensemble} with F1 score {best_f1}, PR AUC {best_pr_auc}, and fitness {best_fitness}")

    # Rank Aggregation (after getting all results)
    test_for_rank = [monte_carlo_ranked_models_F1, monte_carlo_ranked_models_PR,
                     Gan_ranked_by_f1_names, Gan_ranked_by_pr_auc_names,
                     ranked_by_f1_names_sensitivity, ranked_by_pr_auc_names_sensitivity
                     ]

    robust_agg = enhanced_markov_chain_rank_aggregator_text(test_for_rank)
    full_ = [robust_agg[1], thompson_model_names]
    full_aggregated = enhanced_markov_chain_rank_aggregator_text(full_)

    # Write results to file
    directory = f'myresults/robust_aggregated/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(directory, f"robust_aggregated_results_{dataset}_{entity}_{iteration}.txt")

    with open(output_file, 'w') as f:
        f.write("Summary of robust tests:\n")
        f.write("\n1- Monte carlo\n")
        f.write(f"{monte_carlo_ranked_models_F1} \n {monte_carlo_ranked_models_PR} \n")
        f.write("\n Gan\n")
        f.write(f"{Gan_ranked_by_f1_names} \n {Gan_ranked_by_pr_auc_names} \n")
        f.write("\n Off by threshold testing (sensitivity)\n")
        f.write(f"{ranked_by_f1_names_sensitivity} \n {ranked_by_pr_auc_names_sensitivity} \n")
        f.write(f"Robust rank aggregates\n")
        f.write(f"{robust_agg}\n")
        f.write(f"All aggregated\n")
        f.write(f"{full_aggregated}\n")

    return thompson_model_names[0], robust_agg[1], full_aggregated[
        1], best_ensemble, individual_predictions, base_model_predictions_train, base_model_predictions_test, y_true_train, y_true_test, meta_model_type


def find_num_falses(adjusted_y_pred_ind_current, test_data_copy, dataset, entity, values, full_aggregated,
                    best_ensemble, iteration):
    misclassified_current = []
    for predicts in adjusted_y_pred_ind_current:
        true_values = np.array(test_data_copy.entities[0].labels)  # 1 for anomaly, 0 for normal

        predicted_values = np.array(predicts)  # True for predicted anomaly, False for no predicted anomaly

        # Converting boolean predictions to integer for easy plotting (True to 1, False to 0)
        predicted_int = predicted_values.astype(int)

        # Identifying incorrect predictions
        incorrect_predictions = predicted_int != true_values
        misclassified_count = np.sum(incorrect_predictions)  # Number of misclassifications
        misclassified_current.append(misclassified_count)

    misclassified_ensemble = []

    for predicts in [values[3]]:
        true_values = np.array(values[4])  # 1 for anomaly, 0 for normal

        predicted_values = np.array(predicts)  # True for predicted anomaly, False for no predicted anomaly

        # Converting boolean predictions to integer for easy plotting (True to 1, False to 0)
        predicted_int = predicted_values.astype(int)

        # Identifying incorrect predictions
        incorrect_predictions = predicted_int != true_values
        misclassified_count = np.sum(incorrect_predictions)  # Number of misclassifications
        misclassified_ensemble.append(misclassified_count)

    directory = f'myresults/robust_aggregated/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(directory, f"new_robust_aggregated_results_{dataset}_{entity}_{iteration}.txt")

    with open(output_file, 'w') as f:
        f.write("Summary of falses:\n")
        f.write(f"chosen model: {full_aggregated}\n")
        f.write(f'{misclassified_current}\n')
        f.write("Falses for the ensebmle:\n")
        f.write(f"chosen ensemble: {best_ensemble}\n")
        f.write(f'{misclassified_ensemble}')


def run_app(algorithm_list, algorithm_list_instances):
    args = get_args_from_cmdline()
    data_dir = args['dataset_path']
    train_data = load_data(dataset='smd', group='train',
                           entities='machine-3-10', downsampling=10,
                           min_length=256, root_dir=data_dir, normalize=True, verbose=False)
    test_data = load_data(dataset='smd', group='test',
                          entities='machine-3-10', downsampling=10,
                          min_length=256, root_dir=data_dir, normalize=True, verbose=False)

    # Ensure data is correctly loaded
    if not train_data.entities:
        logger.error("Failed to load training data. Please check the dataset and paths.")
        return

    if not test_data.entities:
        logger.error("Failed to load test data. Please check the dataset and paths.")
        return
    entity = 'machine-3-10'
    dataset = 'smd'
    model_trainer = TrainModels(dataset=dataset,
                                entity=entity,
                                algorithm_list=algorithm_list,
                                downsampling=args['downsampling'],
                                min_length=args['min_length'],
                                root_dir=args['dataset_path'],
                                training_size=args['training_size'],
                                overwrite=args['overwrite'],
                                verbose=args['verbose'],
                                save_dir=args['trained_model_path'])
    try:
        model_trainer.train_models(model_architectures=args['model_architectures'])

        # Load trained models

        global trained_models
        trained_models = load_trained_models(algorithm_list_instances, save_dir)

        if not trained_models:
            raise ValueError("No models were loaded. Please check the model paths and ensure models are trained.")

        anomaly_list = ['spikes']

        test_data_before = copy.deepcopy(test_data)
        train_data_before = copy.deepcopy(train_data)
        train_data, anomaliy_sizes_train = Inject(train_data, anomaly_list)
        test_data, anomaly_sizes = Inject(test_data, anomaly_list)

        anomaly_start = np.argmax(test_data.entities[0].labels)
        anomaly_end = test_data.entities[0].Y.shape[1] - np.argmax(test_data.entities[0].labels[::-1])
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20, 6))
        axes[0].plot(test_data.entities[0].Y.flatten(), color='darkblue')
        axes[0].plot(np.arange(anomaly_start, anomaly_end),
                     test_data.entities[0].Y.flatten()[anomaly_start:anomaly_end],
                     color='red')
        axes[0].plot(np.arange(anomaly_start, anomaly_end),
                     test_data_before.entities[0].Y.flatten()[anomaly_start:anomaly_end], color='darkblue',
                     linestyle='--')
        axes[0].set_title('Test data with Injected Anomalies', fontsize=16)
        axes[1].plot(anomaly_sizes.flatten(), color='pink')
        axes[1].plot(test_data.entities[0].labels.flatten(), color='red')
        axes[1].set_title('Anomaly Scores', fontsize=16)
        # Specify the directory
        directory = f'myresults/GA_Ens/{dataset}/{entity}/'
        filename = f'ensemble_scores_{dataset}_{entity}_Data_vs_anomalies_{anomaly_list}.png'
        full_path = os.path.join(directory, filename)

        # Check if the directory exists, and if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the figure
        plt.savefig(full_path, dpi=300)  # Save as PNG file with high resolution

        # plt.show()

        data = test_data_before.entities[0].Y
        targets = test_data_before.entities[0].labels
        mask = test_data_before.entities[0].mask
        iterations = 1
        data_windows, targets_windows, New_mask, num_windows = initialize_sliding_windows(data, targets, mask,
                                                                                          int(np.size(
                                                                                              targets.flatten()) / iterations),
                                                                                          int(np.size(
                                                                                              targets.flatten()) / (
                                                                                                  iterations)) - 5)

        test_data.entities[0].Y = data_windows[0]
        test_data.entities[0].labels = targets_windows[0]
        test_data.entities[0].mask = New_mask[0]
        test_data.entities[0].n_time = np.size(targets_windows[0].flatten())
        test_data.total_time = np.size(targets_windows[0].flatten())
        test_data_new = copy.deepcopy(test_data)
        test_data_new, anomaly_sizes = Inject(test_data_new, anomaly_list)
        best_thompson, robust_agg, full_aggregated, best_ensemble, individual_predictions, base_model_predictions_train, base_model_predictions_test, y_true_train, y_true_test, meta_model_type = run_model_selection_algorithms_1(
            train_data,
            test_data_new, dataset,
            entity, iteration=0)
        i = 1
        # Real-time evaluation
        while i < iterations:
            test_data.entities[0].Y = data_windows[i]
            test_data.entities[0].labels = targets_windows[i]
            test_data.entities[0].mask = New_mask[i]
            test_data.entities[0].n_time = np.size(targets_windows[i].flatten())
            test_data.total_time = np.size(targets_windows[i].flatten())
            trained_models_new = {}

            algorithm_list_new = []
            for model in best_ensemble:
                trained_models_new[model] = trained_models[model]
                algorithm_list_new.append(model)
            test_data_new = copy.deepcopy(test_data)
            test_data_new, anomaly_sizes = Inject(test_data_new, anomaly_list)
            test_data_new_copy = copy.deepcopy(test_data_new)
            individual_predictions, adjusted_y_pred_ind_current, F1_Score_list_ind_curent, PR_AUC_Score_list_ind_curent = evaluate_individual_models(
                [full_aggregated[0]], test_data_new_copy, trained_models)
            test_data_new_copy = copy.deepcopy(test_data_new)
            values = fitness_function(best_ensemble, train_data, test_data_new_copy, trained_models_new,
                                      individual_predictions,
                                      base_model_predictions_train, algorithm_list_instances,
                                      base_model_predictions_test, y_true_train, y_true_test,
                                      meta_model_type=meta_model_type)
            test_data_new_copy = copy.deepcopy(test_data_new)
            find_num_falses(adjusted_y_pred_ind_current, test_data_new_copy, dataset, entity, values,
                            full_aggregated[0],
                            best_ensemble, iteration=i)

            test_data_new_copy = copy.deepcopy(test_data_new)
            best_thompson, robust_agg, full_aggregated, best_ensemble, individual_predictions, base_model_predictions_train, base_model_predictions_test, y_true_train, y_true_test, meta_model_type = run_model_selection_algorithms_1(
                train_data,
                test_data_new_copy, dataset,
                entity, iteration=i)
            i += 1

    except Exception as e:
        logger.info(f'Traceback for Entity: {entity} Dataset: {dataset}')
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    run_app(algorithm_list, algorithm_list_instances)
