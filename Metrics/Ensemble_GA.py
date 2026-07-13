import os
import random
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from Metrics.metrics import prauc, f1_score
from Utils.model_selection_utils import evaluate_model


def _ir_module():
    """Import Explainability.ir, tolerating standalone by-path loading of this
    module where the package root is not on sys.path — falls back to loading
    ir.py directly by its file location."""
    try:
        from Explainability import ir as _ir
        return _ir
    except ModuleNotFoundError:
        import importlib.util
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _spec = importlib.util.spec_from_file_location(
            "explainability_ir", os.path.join(_root, "Explainability", "ir.py"))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        return _mod


def initialize_population(algorithm_list, population_size):
    """
    Initialize the population for the genetic algorithm.

    Args:
        algorithm_list (list): List of available algorithms.
        population_size (int): Desired size of the population.

    Returns:
        list: Initialized population of unique ensembles.
    """
    population = []
    unique_ensembles = set()

    while len(population) < population_size:
        ensemble_size = random.randint(1, len(algorithm_list))
        ensemble = random.sample(algorithm_list, k=ensemble_size)
        ensemble = tuple(sorted(ensemble))  # Canonical ordering and convert to tuple for set operations

        if ensemble not in unique_ensembles and len(ensemble) > 1:
            unique_ensembles.add(ensemble)
            population.append(list(ensemble))  # Convert back to list for the population

    logger.info(f"Initialized population with {population_size} unique ensembles")
    return population


def inject_synthetic_anomalies(y_true):
    """
    Inject synthetic anomalies into the training labels.

    Args:
        y_true (np.ndarray): Array of true labels (1D or 2D with shape (1, N)).

    Returns:
        np.ndarray: Array of true labels with injected anomalies (same shape as input).
    """
    # Handle 2D arrays with batch dimension
    original_shape = y_true.shape
    if y_true.ndim == 2:
        y_true = y_true.flatten()
    
    num_anomalies = int(len(y_true) / 10)
    # num_anomalies = 1
    indices = np.random.choice(len(y_true), num_anomalies, replace=False)
    y_true[indices] = 1
    
    # Restore original shape
    if len(original_shape) == 2:
        y_true = y_true.reshape(original_shape)
    
    return y_true


def train_meta_model(base_model_predictions, y_true):
    """
    Train a logistic regression meta-model.

    Args:
        base_model_predictions (np.ndarray): Predictions from base models.
        y_true (np.ndarray): True labels.

    Returns:
        LogisticRegression: Trained logistic regression meta-model.
    """
    # Clean the data: replace inf/nan values
    base_model_predictions = np.array(base_model_predictions, dtype=np.float32)
    base_model_predictions = np.nan_to_num(base_model_predictions, 
                                           nan=0.0, 
                                           posinf=1.0, 
                                           neginf=0.0)
    
    meta_model = LogisticRegression()
    meta_model.fit(base_model_predictions, y_true)
    logger.info(f"Trained Logistic Regression meta-model")
    return meta_model


def train_meta_model_rf(base_model_predictions, y_true):
    """
    Train a random forest meta-model.

    Args:
        base_model_predictions (np.ndarray): Predictions from base models.
        y_true (np.ndarray): True labels.

    Returns:
        RandomForestClassifier: Trained random forest meta-model.
    """
    # Clean the data: replace inf/nan values
    base_model_predictions = np.array(base_model_predictions, dtype=np.float32)
    base_model_predictions = np.nan_to_num(base_model_predictions, 
                                           nan=0.0, 
                                           posinf=1.0, 
                                           neginf=0.0)
    
    meta_model = RandomForestClassifier()
    meta_model.fit(base_model_predictions, y_true)
    logger.info(f"Trained Random Forest meta-model")
    return meta_model


def train_meta_model_gbm(base_model_predictions, y_true):
    """
    Train a gradient boosting machine meta-model.

    Args:
        base_model_predictions (np.ndarray): Predictions from base models.
        y_true (np.ndarray): True labels.

    Returns:
        GradientBoostingClassifier: Trained gradient boosting machine meta-model.
    """
    # Clean the data: replace inf/nan values
    base_model_predictions = np.array(base_model_predictions, dtype=np.float32)
    base_model_predictions = np.nan_to_num(base_model_predictions, 
                                           nan=0.0, 
                                           posinf=1.0, 
                                           neginf=0.0)
    
    meta_model = GradientBoostingClassifier()
    meta_model.fit(base_model_predictions, y_true)
    logger.info(f"Trained Gradient Boosting Machine meta-model")
    return meta_model


def train_meta_model_svm(base_model_predictions, y_true):
    """
    Train a support vector machine (SVM) meta-model.

    Args:
        base_model_predictions (np.ndarray): Predictions from base models.
        y_true (np.ndarray): True labels.

    Returns:
        SVC: Trained SVM meta-model.
    """
    meta_model = SVC(probability=True)
    meta_model.fit(base_model_predictions, y_true)
    logger.info(f"Trained SVM meta-model")
    return meta_model


def evaluate_model_consistently(data, model, model_name, is_ensemble=False):
    """
    Consistently evaluate a model or ensemble of models on the given data.

    Args:
        data: Dataset for evaluation.
        model: The model or ensemble of models to evaluate.
        model_name (str or list): Name of the model or list of model names for ensemble.
        is_ensemble (bool): Flag indicating if the model is an ensemble.

    Returns:
        tuple: True labels and predictions.
    """
    # Debug: Check raw labels before evaluation
    if hasattr(data, 'entities') and len(data.entities) > 0:
        raw_labels = data.entities[0].labels
        logger.info(f"RAW LABELS in data before evaluation: shape={raw_labels.shape}, unique={np.unique(raw_labels)}, sum={np.sum(raw_labels)}")
    
    y_true_agg_dict = {}
    base_model_predictions_dict = {}
    if is_ensemble:
        y_true_agg = None
        base_model_predictions = []

        for sub_model_name in model_name:
            sub_model = model.get(sub_model_name)
            if sub_model:
                try:
                    evaluation = evaluate_model(data, sub_model, sub_model_name)
                    y_true = evaluation['anomaly_labels'].flatten()
                    y_scores = evaluation['entity_scores'].flatten()
                    
                    # Validate shapes match
                    if len(y_true) != len(y_scores):
                        logger.error(f"Shape mismatch for {sub_model_name}: y_true={len(y_true)}, y_scores={len(y_scores)}")
                        logger.error(f"  Data shape: {data.entities[0].Y.shape}, labels: {data.entities[0].labels.shape}")
                        continue
                    
                    base_model_predictions.append(y_scores)
                    base_model_predictions_dict[sub_model_name] = y_scores
                    if y_true_agg is None:
                        y_true_agg = y_true
                        y_true_agg_dict[sub_model_name] = y_true
                except Exception as e:
                    logger.error(f"Inference failed for {sub_model_name}: {e}")
                    continue
                    
        if len(base_model_predictions) == 0:
            logger.error("No successful model evaluations in ensemble")
            return None, None, {}, {}
            
        base_model_predictions = np.array(base_model_predictions).T
        return y_true_agg, base_model_predictions, y_true_agg_dict, base_model_predictions_dict
    else:
        evaluation = evaluate_model(data, model, model_name)
        y_true = evaluation['anomaly_labels'].flatten()
        y_scores = evaluation['entity_scores'].flatten()

        return y_true, y_scores, y_true_agg_dict, base_model_predictions_dict


def evaluate_individual_models(algorithm_list, test_data, trained_models):
    """
    Evaluate individual models on the test data.

    Args:
        algorithm_list (list): List of algorithm names.
        test_data: Test dataset.
        trained_models (dict): Dictionary of trained models.

    Returns:
        dict: Predictions from individual models.
    """
    predictions = {}
    adjusted_y_pred_list = []
    F1_Score_list = []
    PR_AUC_Score_list = []
    for model_name in algorithm_list:
        model = trained_models.get(model_name)
        if model:
            y_true, y_scores, y_true_agg_dict, y_scores_dict = evaluate_model_consistently(test_data, model, model_name)
            
            # Debug: Check y_true
            logger.info(f"y_true shape: {np.array(y_true).shape}, unique values: {np.unique(y_true)}, sum: {np.sum(y_true)}")
            logger.info(f"y_scores shape: {np.array(y_scores).shape}, min: {np.min(y_scores)}, max: {np.max(y_scores)}, mean: {np.mean(y_scores)}")
            
            # Use range-based metric (segment-aware F1 + PR-AUC) — same as GAN/Borderline/MC
            # robustness tests, which previously produced non-zero F1 on injected data.
            # Strict standard F1 collapses to 0 on synthetic spikes for unsupervised base models,
            # making the single-model branch incomparable to the supervised GA ensemble.
            from Metrics.metrics import range_based_precision_recall_f1_auc
            _, _, best_f1, pr_auc, y_pred_binary = range_based_precision_recall_f1_auc(
                np.asarray(y_true).flatten(), np.asarray(y_scores).flatten()
            )
            
            logger.info(f"Model {model_name}: F1 score = {best_f1}, PR AUC = {pr_auc}")
            predictions[model_name] = (y_true, y_scores)
            adjusted_y_pred_list.append(y_pred_binary)  # Use binary predictions instead of scores
            F1_Score_list.append(best_f1)
            PR_AUC_Score_list.append(pr_auc)
            logger.info(f"First 10 scores for model {model_name}: {y_scores[:10]}")
            logger.info(f"First 10 true labels for model {model_name}: {y_true[:10]}")
    return predictions, adjusted_y_pred_list, F1_Score_list, PR_AUC_Score_list


def fitness_function(ensemble, train_data, test_data, trained_models,
                     individual_predictions,
                     base_model_predictions_train, algorithm_list,
                     base_model_predictions_test, y_true_train, y_true_test,
                     meta_model_type='svm'):
    """
    Evaluate the fitness of an ensemble.

    Args:
        ensemble (list): List of model names in the ensemble.
        train_data: Training dataset.
        test_data: Test dataset.
        trained_models (dict): Dictionary of trained models.
        individual_predictions (list): Predictions from individual base models.
        meta_model_type (str): Type of meta-model to use ('lr', 'rf', 'gbm', 'svm').

    Returns:
        tuple: (best_f1, pr_auc, fitness, y_scores, y_true_test, meta_model).
        The trained meta_model is included so downstream explainability can
        attribute the exact meta-learner without retraining.
    """
    logger.info(f"Evaluating fitness for ensemble: {ensemble}")

    # Sort the ensemble to ensure canonical ordering
    ensemble = sorted(ensemble)

    # Evaluate ensemble on training data

    header_array_train = np.array(algorithm_list)

    # Determine which headers are in the ensemble
    desired_mask_train = np.isin(header_array_train, ensemble)

    # Filter the columns of data array based on the desired headers
    base_model_predictions_train = base_model_predictions_train[:, desired_mask_train]

    # -----

    # Convert the headers to a NumPy array for vectorized operations
    header_array_test = np.array(algorithm_list)

    # Determine which headers are in the ensemble
    desired_mask_test = np.isin(header_array_test, ensemble)

    # Filter the columns of data array based on the desired headers
    base_model_predictions_test = base_model_predictions_test[:, desired_mask_test]
    # Inject synthetic anomalies if the training labels have only one class
    if len(np.unique(y_true_train)) < 2:
        logger.warning(f"Ensemble {ensemble} has only one class in the training labels. Injecting synthetic anomalies.")
        y_true_train = inject_synthetic_anomalies(y_true_train)

    # Train the meta-model based on the specified type
    if meta_model_type == 'lr':
        meta_model = train_meta_model(base_model_predictions_train, y_true_train)
    elif meta_model_type == 'rf':
        meta_model = train_meta_model_rf(base_model_predictions_train, y_true_train)
    elif meta_model_type == 'gbm':
        meta_model = train_meta_model_gbm(base_model_predictions_train, y_true_train)
    elif meta_model_type == 'svm':
        meta_model = train_meta_model_svm(base_model_predictions_train, y_true_train)
    else:
        raise ValueError(f"Unknown meta_model_type: {meta_model_type}")

    # Evaluate ensemble on test data
    # y_true_test, base_model_predictions_test, y_true_test_dict, base_model_predictions_test_dict = evaluate_model_consistently(
    #     test_data, trained_models, ensemble,
    #     is_ensemble=True)

    # Inject synthetic anomalies if the test labels have only one class
    if len(np.unique(y_true_test)) < 2:
        logger.warning(f"Ensemble {ensemble} has only one class in the test labels. Injecting synthetic anomalies.")
        y_true_test = inject_synthetic_anomalies(y_true_test)

    # Clean test predictions: replace inf/nan values before prediction
    base_model_predictions_test = np.array(base_model_predictions_test, dtype=np.float32)
    base_model_predictions_test = np.nan_to_num(base_model_predictions_test, 
                                                 nan=0.0, 
                                                 posinf=1.0, 
                                                 neginf=0.0)

    # Generate prediction scores using the meta-model
    y_scores = meta_model.predict_proba(base_model_predictions_test)[:, 1]

    # Convert probabilities to binary predictions using optimal threshold
    # We need to threshold y_scores for F1 calculation
    thresholds = np.linspace(0.1, 0.9, 50)
    best_threshold = 0.5
    best_temp_f1 = 0
    for threshold in thresholds:
        y_pred_binary = (y_scores >= threshold).astype(int)
        temp_f1, _, _, _, _, _, _ = f1_score(y_pred_binary, y_true_test)
        if temp_f1 > best_temp_f1:
            best_temp_f1 = temp_f1
            best_threshold = threshold
    
    # Use the best threshold for final predictions
    y_pred_binary = (y_scores >= best_threshold).astype(int)
    
    # Calculate evaluation metrics: F1 score and PR AUC
    # _, _, best_f1, pr_auc, adjusted_y_pred = range_based_precision_recall_f1_auc(y_true_test, y_scores)
    best_f1, precision, recall, TP, TN, FP, FN = f1_score(y_pred_binary, y_true_test)
    # best_f1, precision, recall, TP, TN, FP, FN = f1_soft_score(y_scores, y_true_test)
    # best_f1 = get_composite_fscore_raw(y_scores, y_true_test)
    pr_auc = prauc(y_true_test, y_scores)

    # Calculate the fitness score as the average of the F1 score and PR AUC
    # fitness = (best_f1 + pr_auc) / 2
    fitness = best_f1
    logger.info(
        f"Evaluated fitness for ensemble {ensemble} with F1 score {best_f1} and PR AUC {pr_auc}, resulting in fitness {fitness}")
    # The trained meta-model is appended (6th element) so the combination-
    # explainability layer can attribute the EXACT meta-learner the GA built,
    # rather than retraining one. Existing callers index [0..4] and are unaffected.
    return best_f1, pr_auc, fitness, y_scores, y_true_test, meta_model


def selection(population, fitness_scores, num_selected):
    """
    Select the top ensembles based on fitness scores.

    Args:
        population (list): List of ensembles.
        fitness_scores (list): List of fitness scores corresponding to the population.
        num_selected (int): Number of ensembles to select.

    Returns:
        list: Selected top ensembles.
    """
    selected_indices = np.argsort(fitness_scores)[-num_selected:]
    selected = [population[i] for i in selected_indices]
    logger.info(f"Selected top {num_selected} ensembles with scores {fitness_scores}")
    return selected


def crossover(parent1, parent2):
    """
    Perform crossover between two parent ensembles to create a child ensemble.

    Args:
        parent1 (list): First parent ensemble.
        parent2 (list): Second parent ensemble.

    Returns:
        list: Child ensemble resulting from the crossover.
    """
    crossover_point1 = random.randint(1, len(parent1))
    crossover_point2 = random.randint(1, len(parent2))
    child = parent1[:crossover_point1] + parent2[crossover_point2:]
    child = list(set(child))
    child = sorted(child)
    logger.info(f"Crossover parents {parent1} and {parent2} to create child {child}")
    return child


def mutate(ensemble, mutation_rate, algorithm_list):
    """
    Perform mutation on an ensemble.

    Args:
        ensemble (list): Ensemble to mutate.
        mutation_rate (float): Mutation rate.
        algorithm_list (list): List of available algorithms.

    Returns:
        list: Mutated ensemble.
    """
    mutated_ensemble = ensemble.copy()
    for i in range(len(mutated_ensemble)):
        if random.random() < mutation_rate:
            available_models = [model for model in algorithm_list if model not in mutated_ensemble]
            if available_models:
                original_model = mutated_ensemble[i]
                mutated_ensemble[i] = random.choice(available_models)
                logger.info(f"Mutated model {original_model} to {mutated_ensemble[i]} in ensemble {ensemble}")
            else:
                logger.warning(f"No available models to mutate in the ensemble: {mutated_ensemble}")

    if random.random() < mutation_rate:
        if len(mutated_ensemble) > 3 and random.random() > 0.5:
            model_to_remove = random.choice(mutated_ensemble)
            mutated_ensemble.remove(model_to_remove)
            logger.info(f"Removed model {model_to_remove} from ensemble {ensemble}")
        else:
            possible_models = [model for model in algorithm_list if model not in mutated_ensemble]
            if possible_models:
                model_to_add = random.choice(possible_models)
                mutated_ensemble.append(model_to_add)
                logger.info(f"Added model {model_to_add} to ensemble {ensemble}")
            else:
                logger.warning(f"No available models to add to the ensemble: {mutated_ensemble}")

    mutated_ensemble = sorted(mutated_ensemble)  # Ensure canonical ordering after mutation
    return mutated_ensemble


def plot_scores_vs_true(data, F1_Score_list, PR_AUC_Score_list, adjusted_y_pred, list_ensemble, plot_name, plot_path):
    max_arg_f1 = np.argmax(np.array(F1_Score_list))
    max_arg_pr_auc = np.argmax(np.array(PR_AUC_Score_list))
    print(data.entities[0].labels)
    print(adjusted_y_pred[max_arg_f1])
    true_values = np.array(data.entities[0].labels)  # 1 for anomaly, 0 for normal
    print(10 * '=')
    print(true_values)
    predicted_values = np.array(
        adjusted_y_pred[max_arg_f1])  # True for predicted anomaly, False for no predicted anomaly

    # Converting boolean predictions to integer for easy plotting (True to 1, False to 0)
    predicted_int = predicted_values.astype(int)
    best_ = list_ensemble[max_arg_f1]
    # Identifying incorrect predictions
    incorrect_predictions = predicted_int != true_values
    misclassified_count = np.sum(incorrect_predictions)  # Number of misclassifications
    total_anomalies = np.sum(true_values)  # Total number of real anomalies
    total_data = len(true_values)  # Total number of data points

    # Highlight incorrect predictions with a different marker
    print(incorrect_predictions)
    if incorrect_predictions.ndim == 2:
        pass
    else:

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(true_values, '.', label='True Values (Anomalies)', color='blue')  # Plot true values
        # plt.plot(predicted_int, 'x', label='Predicted Values (Anomalies)', color='red')  # Plot predicted values
        plt.scatter(np.where(incorrect_predictions)[0], predicted_int[incorrect_predictions], facecolors='none',
                    edgecolors='purple', s=100, label='Incorrect Predictions', linewidth=2)
        plt.title(
            f'True vs. Predicted Anomalies \n Misclassified Anomalies: {misclassified_count}, {best_}\n Total Anomalies: {total_anomalies} \n Total Data: {total_data}')
        plt.xlabel('Index')
        plt.ylabel('Anomaly Presence')
        plt.yticks([0, 1], ['No Anomaly', 'Anomaly'])  # Set y-ticks to be explicit about what 0 and 1 represent
        plt.legend()
        plt.grid(True)
        # Specify the directory
        directory = plot_path
        filename = plot_name
        full_path = os.path.join(directory, filename)

        # Get the directory path for the file
        output_dir = os.path.dirname(full_path)

        # Create all directories in the path if they don't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the figure
        # plt.savefig(full_path, dpi=300)  # Save as PNG file with high resolution

        # plt.show()


def plot_models_scores(algorithm_list, test_data, y_scores_list, dataset, entity, F1_Score_list_ind_curent,
                       PR_AUC_Score_list_ind_curent):
    data = test_data.entities[0].Y
    targets = test_data.entities[0].labels

    # Ensure unique algorithms and corresponding values
    unique_algorithms = []
    unique_y_scores_list = []
    unique_F1_Score_list = []
    unique_PR_AUC_Score_list = []

    seen = set()
    for i, algorithm in enumerate(algorithm_list):
        if algorithm not in seen:
            seen.add(algorithm)
            unique_algorithms.append(algorithm)
            unique_y_scores_list.append(y_scores_list[i])
            unique_F1_Score_list.append(F1_Score_list_ind_curent[i])
            unique_PR_AUC_Score_list.append(PR_AUC_Score_list_ind_curent[i])

    # Determine the number of rows needed
    num_algorithms = len(unique_algorithms)
    num_rows = 2 + num_algorithms  # 2 for original data and labels, rest for each algorithm

    # Plot the data
    fig, axes = plt.subplots(num_rows, 1, figsize=(18, 4 * num_rows), sharex=True)

    # First row: plot the data
    axes[0].plot(data.flatten(), label='Data', color='blue')
    axes[0].set_title('Data')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True)

    # Second row: plot the labels with spikes
    axes[1].plot(targets, label='Labels', color='gray')
    spike_indices = np.where(targets == 1)[0]
    spike_values = np.ones_like(spike_indices)  # Set spikes at 1 for visibility
    axes[1].vlines(spike_indices, ymin=0, ymax=spike_values, color='red', label='Anomalies')
    axes[1].set_title('Labels')
    axes[1].set_ylabel('Label')
    axes[1].grid(True)

    # Loop over the unique y_scores_list and plot each under the original labels
    for i, algorithm in enumerate(unique_algorithms):
        y_scores = unique_y_scores_list[i]
        f1_score_value = unique_F1_Score_list[i]
        pr_auc_value = unique_PR_AUC_Score_list[i]

        # Plot the y_scores
        axes[i + 2].plot(y_scores, label=f'{algorithm} Scores', color='gray')

        # Identify spikes, true positives, false positives, and false negatives
        spike_indices = np.where(y_scores >= 0.5)[0]
        true_positive_indices = np.intersect1d(spike_indices, np.where(targets == 1)[0])
        false_positive_indices = np.setdiff1d(spike_indices, true_positive_indices)
        false_negative_indices = np.setdiff1d(np.where(targets == 1)[0], true_positive_indices)

        # Plot detected anomalies
        axes[i + 2].vlines(spike_indices, ymin=0, ymax=1, color='red', label='Detected Anomalies')

        # Highlight true positives with a different color
        # axes[i + 2].vlines(true_positive_indices, ymin=0, ymax=1, color='green', label='True Positives')

        # Highlight false positives with a different color
        # axes[i + 2].vlines(false_positive_indices, ymin=0, ymax=1, color='orange', label='False Positives')

        # Highlight false negatives with a different color
        # axes[i + 2].vlines(false_negative_indices, ymin=0, ymax=1, color='purple', label='False Negatives')

        axes[i + 2].set_title(f'{algorithm} Anomaly Scores, F1 Score = {f1_score_value}, PR AUC = {pr_auc_value}')
        axes[i + 2].set_ylabel('Score')
        axes[i + 2].grid(True)

    # Add legend to the last axis
    handles, labels = axes[1].get_legend_handles_labels()
    detected_handles, detected_labels = axes[2].get_legend_handles_labels()
    combined_handles = handles + detected_handles
    combined_labels = labels + detected_labels
    fig.legend(combined_handles, combined_labels, loc='upper right')

    axes[-1].set_xlabel('Time (index)')

    plt.tight_layout()
    directory = f'myresults/Thomposon/{dataset}/{entity}/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    #plt.savefig(f'{directory}/performance_plot_nigg.png')
    # plt.show()


def genetic_algorithm(dataset, entity, train_data, test_data, algorithm_list, trained_models, meta_model_type,
                      population_size, generations, mutation_rate, explain: bool = False):
    """
    Run the genetic algorithm to find the best ensemble of models.

    Args:
        dataset (str): Dataset name.
        entity (str): Entity name.
        train_data: Training dataset.
        test_data: Test dataset.
        algorithm_list (list): List of algorithm names.
        trained_models (dict): Dictionary of trained models.
        meta_model_type (str): Type of meta-model to use ('lr', 'rf', 'gbm', 'svm').
        population_size (int): Size of the population.
        generations (int): Number of generations.
        mutation_rate (float): Mutation rate.

    Returns:
        tuple: Best ensemble, best F1 score, best PR AUC, and best fitness score.
    """
    # mevaluation_instance = Mevaluation()
    # mevaluation_instance.evaluate_model(train_data, test_data)  # Evaluate individual models before GA

    individual_predictions, adjusted_y_pred_ind, F1_Score_list_ind, PR_AUC_Score_list_ind = evaluate_individual_models(
        algorithm_list, test_data, trained_models)
    logger.info(f"  ✓ Individual model evaluation complete ({len(algorithm_list)} models)")
    
    # Skip plotting to save time (but keep path definitions for later use)
    # logger.info(f"  → Plotting individual model scores...")
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_name = f'ensemble_scores_{dataset}_{entity}_{meta_model_type}_{population_size}_{generations}_{mutation_rate}_UMS_{date_time_string}.png'
    plot_path = f'myresults/Outputs/GA_Ens/{dataset}/{entity}'
    # plot_scores_vs_true(test_data, F1_Score_list_ind, PR_AUC_Score_list_ind, adjusted_y_pred_ind, algorithm_list,
    #                     plot_name, plot_path)
    # logger.info(f"  ✓ Plot saved to {plot_path}/{plot_name}")
    
    logger.info(f"  → Initializing GA population (size={population_size})...")
    import time as time_module
    start_init = time_module.time()
    # individual_predictions = []
    population = initialize_population(algorithm_list, population_size)
    logger.info(f"  ✓ Population initialized in {time_module.time() - start_init:.2f}s")
    print(population)
    evaluated_ensembles = {}  # HashMap to track evaluated ensembles and their scores
    file_name = f'myresults/Outputs/GA_Ens/ensemble_scores_{dataset}_{entity}_{meta_model_type}_{population_size}_{generations}_{mutation_rate}_{date_time_string}.txt'

    best_f1 = 0
    best_pr_auc = 0
    best_fitness = 0
    best_ensemble = None
    best_meta_model = None   # the trained meta-model of the best ensemble (for combination explainability)
    adjusted_y_pred_list = []
    F1_Score_list = []
    list_ensemble = []
    PR_AUC_Score_list = []
    fitness_list = []
    # Per-generation populations — used by the selection-explainability layer
    # (Axis 3: evolutionary survival rate). Snapshotted at the top of each loop
    # iteration, before fitness evaluation, so it reflects the population OF
    # that generation rather than the next one's offspring.
    generation_populations: List[List[List[str]]] = []
    
    # Prepare training data for meta-model
    logger.info(f"  → Evaluating all {len(algorithm_list)} models on TRAINING data (for meta-model training)...")
    start_train = time_module.time()
    y_true_train, base_model_predictions_train, y_true_train_dict, base_model_predictions_train_dict = evaluate_model_consistently(
        train_data,
        trained_models,
        algorithm_list,
        is_ensemble=True)
    logger.info(f"  ✓ Training data evaluation complete in {time_module.time() - start_train:.2f}s")
    
    # Reuse test predictions from individual_predictions instead of re-computing
    logger.info(f"  → Reusing test predictions from individual evaluation (no re-computation)...")
    start_reuse = time_module.time()
    y_true_test = None
    base_model_predictions_test_list = []
    for model_name in algorithm_list:
        if model_name in individual_predictions:
            y_true, y_scores = individual_predictions[model_name]
            base_model_predictions_test_list.append(y_scores)
            if y_true_test is None:
                y_true_test = y_true
    base_model_predictions_test = np.array(base_model_predictions_test_list).T
    logger.info(f"  ✓ Test predictions reused in {time_module.time() - start_reuse:.4f}s (vs ~{start_train - start_train:.2f}s if recomputed)")
    
    logger.info(f"  ✓ Meta-model preparation complete. Starting GA generations...")
    print("y_true_train mine")
    print(y_true_train)
    for generation in range(generations):
        logger.info(f"Generation {generation + 1}")
        print(f"Generation {generation + 1}")

        if explain:
            generation_populations.append([list(ind) for ind in population if ind is not None])

        fitness_results = []
        for ensemble in population:
            if ensemble is not None:  # Ensure ensemble is not None
                ensemble_key = tuple(sorted(ensemble))  # Create a unique key for the ensemble
                if ensemble_key not in evaluated_ensembles:
                    fitness_result = fitness_function(ensemble, train_data, test_data, trained_models,
                                                      individual_predictions,
                                                      base_model_predictions_train, algorithm_list,
                                                      base_model_predictions_test, y_true_train, y_true_test,
                                                      meta_model_type=meta_model_type)
                    evaluated_ensembles[ensemble_key] = fitness_result


                else:
                    fitness_result = evaluated_ensembles[ensemble_key]
                adjusted_y_pred = fitness_result[3]
                adjusted_y_pred_list.append(adjusted_y_pred)
                F1_Score_list.append(fitness_result[0])
                list_ensemble.append(ensemble_key)
                PR_AUC_Score_list.append(fitness_result[1])
                fitness_list.append(fitness_result[2])
                fitness_results.append(fitness_result)

        fitness_scores = [result[2] for result in fitness_results]
        f1_scores = [result[0] for result in fitness_results]
        pr_aucs = [result[1] for result in fitness_results]

        print(f"Fitness Scores: {fitness_scores}")

        selected = selection(population, fitness_scores, max(1, population_size // 2))
        new_population = selected.copy()

        while len(new_population) < population_size:
            if len(selected) > 1:
                parent1, parent2 = random.sample(selected, 2)
                child = crossover(parent1, parent2)
            else:
                child = selected[0]
            child = mutate(child, mutation_rate, algorithm_list)
            new_population.append(child)

        

        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > best_fitness:
            best_f1 = f1_scores[best_idx]
            best_pr_auc = pr_aucs[best_idx]
            best_fitness = fitness_scores[best_idx]
            best_ensemble = population[best_idx]
            # Capture the meta-model that achieved this best fitness (index 5 of the
            # fitness tuple) for the combination-explainability layer. Aligned with
            # f1_scores / fitness_scores, which are all derived from fitness_results.
            best_meta_model = fitness_results[best_idx][5]
        population = new_population

        logger.info(f"End of Generation {generation + 1}, Population: {population}")
        print(f"End of Generation {generation + 1}, Population: {population}")

    misclassified_ens = []
    for predicts in adjusted_y_pred_list:
        true_values = np.array(test_data.entities[0].labels)  # 1 for anomaly, 0 for normal

        predicted_values = np.array(predicts)  # True for predicted anomaly, False for no predicted anomaly

        # Converting boolean predictions to integer for easy plotting (True to 1, False to 0)
        predicted_int = predicted_values.astype(int)

        # Identifying incorrect predictions
        incorrect_predictions = predicted_int != true_values
        misclassified_count = np.sum(incorrect_predictions)  # Number of misclassifications
        misclassified_ens.append(misclassified_count)
    directory = plot_path
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(directory, f"GA_Differeneces_results_{dataset}_{entity}.txt")
    #with open(output_file, 'w') as f:
     #   f.write("All Ensembles:\n")
      #  f.write(f" Evaluated Ensembles: {evaluated_ensembles.keys()}\n")
       # f.write("All Differences:\n")
        #f.write(f"{misclassified_ens}")
    ensemble_names = [name for name in evaluated_ensembles.keys()]
    f1_scores = [result[0] for result in evaluated_ensembles.values()]
    pr_auc_scores = [result[1] for result in evaluated_ensembles.values()]
    flat_ensemble_names = ['_'.join(names) for names in ensemble_names]
    plot_name = f'myresults/Outputs/GA_Ens/ensemble_scores_{dataset}_{entity}_{meta_model_type}_{population_size}_{generations}_{mutation_rate}_ensemble_{date_time_string}.png'
    plot_models_scores(list_ensemble, test_data, adjusted_y_pred_list, dataset, entity, F1_Score_list,
                       PR_AUC_Score_list)
    plot_scores_vs_true(test_data, F1_Score_list, PR_AUC_Score_list, adjusted_y_pred_list, list_ensemble, plot_name,
                        plot_path)
    # Plot for F1 scores
    plt.figure(figsize=(10, 5))
    plt.plot(flat_ensemble_names, f1_scores, marker='o', linestyle='-', color='b')
    plt.title('F1 Scores of Ensembles')
    plt.xlabel('Ensemble Name')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)  # Rotating the x-axis labels for better readability
    plt.grid(True)
    plt.tight_layout()
    # Specify the directory
    directory = plot_path
    filename = f'ensemble_scores_{dataset}_{entity}_{meta_model_type}_{population_size}_{generations}_{mutation_rate}_F1_{date_time_string}.png'
    full_path = os.path.join(directory, filename)

    # Check if the directory exists, and if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the figure
    # plt.savefig(full_path, dpi=300)  # Save as PNG file with high resolution

    # plt.show()
    #
    # # Plot for PR_AUC scores
    plt.figure(figsize=(10, 5))
    plt.plot(flat_ensemble_names, pr_auc_scores, marker='o', linestyle='-', color='r')
    plt.title('PR_AUC Scores of Ensembles')
    plt.xlabel('Ensemble Name')
    plt.ylabel('PR AUC Score')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    # Specify the directory
    directory = plot_path
    filename = f'ensemble_scores_{dataset}_{entity}_{meta_model_type}_{population_size}_{generations}_{mutation_rate}_PR_{date_time_string}.png'
    full_path = os.path.join(directory, filename)

    # Check if the directory exists, and if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the figure
    # plt.savefig(full_path, dpi=300)  # Save as PNG file with high resolution

    # plt.show()
    logger.info(
        f"Best ensemble found: {best_ensemble} with F1 score {best_f1}, PR AUC {best_pr_auc}, and fitness {best_fitness}")
    print(
        f"Best ensemble found: {best_ensemble} with F1 score {best_f1}, PR AUC {best_pr_auc}, and fitness {best_fitness}")
    # Sort evaluated_ensembles by fitness score before writing to the file
    sorted_ensembles = sorted(evaluated_ensembles.items(), key=lambda x: x[1][2], reverse=True)
    # Save the results to a text file

   # with open(file_name, "w") as f:
    #    for ensemble, result in sorted_ensembles:
     #       f.write(f"Ensemble: {list(ensemble)}, f1 : {result[0]}, PR_AUC: {result[1]}, Fitness Score: {result[2]}\n")

    if explain and best_ensemble:
        def _evaluate_fitness(subset):
            key = tuple(sorted(subset))
            if key in evaluated_ensembles:
                return evaluated_ensembles[key][2]
            res = fitness_function(list(subset), train_data, test_data, trained_models,
                                   individual_predictions, base_model_predictions_train,
                                   algorithm_list, base_model_predictions_test,
                                   y_true_train, y_true_test,
                                   meta_model_type=meta_model_type)
            evaluated_ensembles[key] = res
            return res[2]

        explain_ga_selection(best_ensemble, evaluated_ensembles, generation_populations,
                             algorithm_list, population_size, _evaluate_fitness,
                             dataset, entity, explain=True)

        explain_ga_combination(best_ensemble, algorithm_list,
                               base_model_predictions_train, base_model_predictions_test,
                               y_true_train, y_true_test, meta_model_type,
                               dataset, entity, meta_model=best_meta_model, explain=True)

    return best_ensemble, best_f1, best_pr_auc, best_fitness, individual_predictions, base_model_predictions_train, base_model_predictions_test, y_true_train, y_true_test, meta_model_type

# Usage
# Assuming train_data and test_data are already loaded and preprocessed
# algorithm_list = ['LOF', 'NN', 'RNN']
# trained_models = {'LOF': lof_model, 'NN': nn_model, 'RNN': rnn_model}
# best_ensemble, best_f1, best_pr_auc, best_fitness = genetic_algorithm(train_data, test_data, algorithm_list, trained_models, meta_model_type='lr')
# You can change meta_model_type to 'rf', 'gbm', or 'svm' for Random Forest, Gradient Boosting Machine, and SVM respectively


# ════════════════════════════════════════════════════════════════════════════
#  GA Ensemble Selection Explainability
#
#  Three analytical axes per candidate detector:
#    1. Utility       — LOFO marginal fitness change on the best ensemble,
#                       and mean marginal contribution across evaluated subsets.
#    2. Complementarity — interaction matrix I_jk over detector pairs:
#                       I_jk = E[y | d_j=1, d_k=1] - (E[y|d_j=1] + E[y|d_k=1]),
#                       with y = subset fitness.
#    3. Stability     — evolutionary survival rate per generation:
#                       P(d_j, g) = (#individuals in gen g containing d_j) /
#                                    population_size.
# ════════════════════════════════════════════════════════════════════════════


def _conditional_mean_fitness(
    evaluated_ensembles: Dict[Tuple[str, ...], tuple],
    present: Tuple[str, ...] = (),
    absent: Tuple[str, ...] = (),
) -> Tuple[float, int]:
    """Mean of subset fitness over evaluated subsets that contain every detector
    in `present` and none in `absent`. Returns (mean, count). When count == 0
    the mean is NaN."""
    present_set = set(present)
    absent_set = set(absent)
    vals = []
    for key, result in evaluated_ensembles.items():
        members = set(key)
        if present_set.issubset(members) and absent_set.isdisjoint(members):
            vals.append(float(result[2]))   # result = (f1, pr_auc, fitness, ...)
    if not vals:
        return float('nan'), 0
    return float(np.mean(vals)), len(vals)


def compute_lofo_utility(
    best_ensemble: List[str],
    evaluate_fitness: Callable[[List[str]], float],
) -> Dict[str, float]:
    """
    Axis 1a — LOFO marginal fitness change on the final ensemble.

    For each detector d in best_ensemble:
        marginal[d] = fitness(best_ensemble) − fitness(best_ensemble \\ {d})
    Positive = removing d hurt fitness (d was pulling weight).
    NaN if best_ensemble has fewer than 2 detectors (LOFO undefined).
    """
    if not best_ensemble or len(best_ensemble) < 2:
        return {d: float('nan') for d in (best_ensemble or [])}
    base = float(evaluate_fitness(list(best_ensemble)))
    out: Dict[str, float] = {}
    for d in best_ensemble:
        reduced = [x for x in best_ensemble if x != d]
        out[d] = base - float(evaluate_fitness(reduced))
    return out


def compute_mean_marginal_contribution(
    evaluated_ensembles: Dict[Tuple[str, ...], tuple],
    algorithm_list: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Axis 1b — mean marginal contribution across evaluated subsets.

    For each detector d:
        contribution[d] = E[fitness | d present] − E[fitness | d absent]
    over the distinct subsets the GA evaluated. Returns a dict per detector:
        {'contribution', 'e_present', 'e_absent', 'n_present', 'n_absent'}.
    Missing means are NaN with the corresponding count = 0.
    """
    out: Dict[str, Dict[str, float]] = {}
    for d in algorithm_list:
        e_p, n_p = _conditional_mean_fitness(evaluated_ensembles, present=(d,))
        e_a, n_a = _conditional_mean_fitness(evaluated_ensembles, absent=(d,))
        contrib = (e_p - e_a) if (n_p > 0 and n_a > 0) else float('nan')
        out[d] = {
            'contribution': contrib,
            'e_present': e_p,
            'e_absent': e_a,
            'n_present': n_p,
            'n_absent': n_a,
        }
    return out


def _build_subset_matrix(
    evaluated_ensembles: Dict[Tuple[str, ...], tuple],
    algorithm_list: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the binary inclusion matrix X (n_subsets × d) and fitness vector y from
    the evaluated subsets. X[i, c] = 1 iff algorithm_list[c] is in subset i.
    y[i] = fitness of subset i (element [2] of the cached fitness tuple).
    """
    d = len(algorithm_list)
    col = {name: c for c, name in enumerate(algorithm_list)}
    rows, ys = [], []
    for key, result in evaluated_ensembles.items():
        z = np.zeros(d, dtype=float)
        for name in key:
            if name in col:
                z[col[name]] = 1.0
        rows.append(z)
        ys.append(float(result[2]))
    if not rows:
        return np.zeros((0, d)), np.zeros(0)
    return np.array(rows, dtype=float), np.array(ys, dtype=float)


# NOTE: The Complementarity axis (Friedman H) is currently DISABLED in
# explain_ga_selection. This function and its plots (plot_ga_interaction,
# plot_ga_total_interaction) are retained, but not invoked, so the axis can be
# re-enabled by uncommenting the call sites in explain_ga_selection and restoring
# the complementarity inputs to classify_detector_archetypes / plot_ga_archetypes.
def compute_friedman_h(
    evaluated_ensembles: Dict[Tuple[str, ...], tuple],
    algorithm_list: List[str],
    surrogate: str = "rf",
    predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    max_reference: int = 500,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Axis 2 — Friedman & Popescu (2008) H-statistic over the GA-explored subset
    space (https://projecteuclid.org/journals/annals-of-applied-statistics/volume-2/issue-3/Predictive-learning-via-rule-ensembles/10.1214/07-AOAS148.pdf).

    Subset fitness is treated as a function F(z) of the binary detector-inclusion
    vector z ∈ {0,1}^d. A surrogate regressor F̂ is fit on the evaluated
    (z, fitness) pairs, then partial-dependence functions of F̂ yield:

      Two-way interaction (Eq. 44):
        H²_jk = Σ_i (F̃_jk − F̃_j − F̃_k)² / Σ_i F̃_jk²
      Total interaction (Eq. 45):
        H²_j  = Σ_i (F̃ − F̃_j − F̃_{\\j})² / Σ_i F̃²

    where ~ denotes mean-centering over the reference set and H = sqrt(max(0,H²)).
    Because inputs are binary, every partial-dependence function takes ≤2 (single)
    or ≤4 (pair) distinct values, so the computation is cheap.

    Parameters
    ----------
    surrogate : "rf" | "gbm"          Surrogate family (used only if predict_fn is None).
    predict_fn : callable | None      Inject a fitted F̂: (m×d array) -> (m,) preds.
                                      When None, a surrogate is fit on (X, y).
    max_reference : int               Cap on reference rows (subsampled if exceeded).

    Returns
    -------
    dict with keys:
        "H_two_way"   : {(j, k): H_jk}  symmetric (j != k); NaN where undefined
        "H_total"     : {j: H_j}        per detector; NaN where undefined
        "n_subsets"   : int             training subsets used
        "surrogate"   : str
        "surrogate_r2": float           in-sample R² (NaN when predict_fn injected)
        "feasible"    : bool            False when too few/degenerate subsets
    """
    d = len(algorithm_list)
    nan_two = {(j, k): float('nan')
               for i, j in enumerate(algorithm_list) for k in algorithm_list[i + 1:]}
    # make symmetric
    for (j, k) in list(nan_two.keys()):
        nan_two[(k, j)] = float('nan')
    nan_tot = {j: float('nan') for j in algorithm_list}

    X, y = _build_subset_matrix(evaluated_ensembles, algorithm_list)
    n = X.shape[0]
    base = {"H_two_way": nan_two, "H_total": nan_tot, "n_subsets": n,
            "surrogate": surrogate, "surrogate_r2": float('nan'), "feasible": False}
    if n < 3 or np.allclose(y, y[0]):
        return base

    # ── Surrogate F̂ ───────────────────────────────────────────────────────
    surrogate_r2 = float('nan')
    if predict_fn is None:
        try:
            if surrogate == "gbm":
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(random_state=random_state)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=200, random_state=random_state)
            model.fit(X, y)
            preds_train = model.predict(X)
            ss_res = float(np.sum((y - preds_train) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            surrogate_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            predict = model.predict
        except Exception as e:
            logger.warning(f"Friedman H surrogate fit failed: {e}")
            return base
    else:
        predict = predict_fn

    # ── Reference set (subsample if large) ─────────────────────────────────
    Xr = X
    if n > max_reference:
        rng = np.random.RandomState(random_state)
        Xr = X[rng.choice(n, size=max_reference, replace=False)]
    N = Xr.shape[0]

    def _predict_with(col_overrides: Dict[int, float]) -> np.ndarray:
        Z = Xr.copy()
        for c, v in col_overrides.items():
            Z[:, c] = v
        return np.asarray(predict(Z), dtype=float)

    def _center(a: np.ndarray) -> np.ndarray:
        return a - float(np.mean(a))

    # Single-variable PD predictions with coord c overridden: pred1[c][v] is the
    # length-N vector F̂(Xr with col c := v). Used two ways below:
    #   • its MEAN is the PD scalar F_c(v) = (1/N) Σ_m F̂(x_m, c:=v);
    #   • the per-row vectors feed the all-but-c PD (Eq. 45).
    pred1 = {c: {0.0: _predict_with({c: 0.0}), 1.0: _predict_with({c: 1.0})}
             for c in range(d)}
    p_marg = {c: float(np.mean(Xr[:, c])) for c in range(d)}   # marginal P(z_c=1)
    f_base = np.asarray(predict(Xr), dtype=float)
    f_tilde = _center(f_base)
    f_base_sq = float(np.sum(f_tilde ** 2))

    # PD scalars F_c(v), then the centered single-variable PD evaluated at each
    # row's own coordinate x_ic.
    pd1_scalar = {c: {0.0: float(np.mean(pred1[c][0.0])),
                      1.0: float(np.mean(pred1[c][1.0]))}
                  for c in range(d)}
    pd_j = {}
    for c in range(d):
        vals = np.where(Xr[:, c] == 1.0, pd1_scalar[c][1.0], pd1_scalar[c][0.0])
        pd_j[c] = _center(vals)

    # ── Two-way H (Eq. 44) ─────────────────────────────────────────────────
    H_two: Dict[Tuple[str, str], float] = {}
    for a in range(d):
        for b in range(a + 1, d):
            # Pairwise PD scalars F_ab(va, vb) = mean over rows of F̂(Xr, a:=va, b:=vb).
            pd2_scalar = {}
            for va in (0.0, 1.0):
                for vb in (0.0, 1.0):
                    pd2_scalar[(va, vb)] = float(np.mean(_predict_with({a: va, b: vb})))
            # Evaluate the pairwise PD at each row's own (x_ia, x_ib).
            pjk = np.array([pd2_scalar[(Xr[i, a], Xr[i, b])] for i in range(N)], dtype=float)
            f_jk = _center(pjk)
            num = float(np.sum((f_jk - pd_j[a] - pd_j[b]) ** 2))
            den = float(np.sum(f_jk ** 2))
            h = float(np.sqrt(max(0.0, num / den))) if den > 0 else float('nan')
            H_two[(algorithm_list[a], algorithm_list[b])] = h
            H_two[(algorithm_list[b], algorithm_list[a])] = h

    # ── Total H (Eq. 45) ───────────────────────────────────────────────────
    H_tot: Dict[str, float] = {}
    for c in range(d):
        # PD on all-but-c: average F̂ over z_c drawn from its marginal.
        pd_not = (1.0 - p_marg[c]) * pred1[c][0.0] + p_marg[c] * pred1[c][1.0]
        f_not = _center(pd_not)
        num = float(np.sum((f_tilde - pd_j[c] - f_not) ** 2))
        h = float(np.sqrt(max(0.0, num / f_base_sq))) if f_base_sq > 0 else float('nan')
        H_tot[algorithm_list[c]] = h

    return {"H_two_way": H_two, "H_total": H_tot, "n_subsets": n,
            "surrogate": "injected" if predict_fn is not None else surrogate,
            "surrogate_r2": surrogate_r2, "feasible": True}


def compute_survival_rates(
    generation_populations: List[List[List[str]]],
    algorithm_list: List[str],
    population_size: int,
) -> Dict[str, List[float]]:
    """
    Axis 3 — survival rate per detector per generation:
        P(d, g) = (count of individuals in generation g containing d) / population_size
    Returns {detector: [P(d, g) for g in generations]}.
    """
    out: Dict[str, List[float]] = {d: [] for d in algorithm_list}
    denom = float(population_size) if population_size > 0 else 1.0
    for pop in generation_populations:
        for d in algorithm_list:
            count = sum(1 for ind in pop if d in ind)
            out[d].append(count / denom)
    return out


# ── Functional archetypes (intersection of the axes) ────────────────────────

# Complementarity (Friedman H) is currently disabled, so each detector is labelled
# by the (Utility, Stability) high/low pair as a 2-letter H/L code, e.g. "HL" =
# high utility, low stability. A detector with no utility data is "Unclassified".
ARCHETYPE_UNCLASSIFIED = "Unclassified"

ARCHETYPE_ORDER = [
    "HH", "HL", "LH", "LL",
    ARCHETYPE_UNCLASSIFIED,
]


def _assign_archetype(u_high: bool, s_high: bool, util_nan: bool) -> str:
    """
    Label the (utility, stability) high/low pair as a 2-letter H/L code (e.g.
    "HL"). A detector with no utility data is "Unclassified".
    (Complementarity axis is currently disabled — see compute_friedman_h note.)
    """
    if util_nan:
        return ARCHETYPE_UNCLASSIFIED
    return ("H" if u_high else "L") + ("H" if s_high else "L")


def _finite_median(values: List[float]) -> float:
    """Median over the finite (non-NaN) values; NaN if none are finite."""
    finite = [v for v in values if not np.isnan(v)]
    return float(np.median(finite)) if finite else float('nan')


def classify_detector_archetypes(
    mean_marginal: Dict[str, Dict[str, float]],
    survival: Dict[str, List[float]],
    algorithm_list: List[str],
    abs_utility: float = 0.0,
    abs_stability: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """
    Classify each detector into a functional archetype from the intersection of
    the two active axes (Utility × Stability). Reports BOTH a relative
    (median-split) and an absolute (fixed-cutoff) scheme side by side.
    (Complementarity / Friedman H is currently disabled.)

    Axis scalars (per detector):
      utility        = mean_marginal[d]['contribution']  (Axis 1b only; LOFO excluded)
      stability_mean = mean(survival[d])  (the Stability axis)

    A detector is "stable-high" when its mean survival is above the threshold.
    (stability_trend = P(last) − P(first) is still reported for context, but does
    not affect the classification.)

    Returns {detector: {utility, stability_mean, stability_trend,
                        "relative": {u_high,s_high,archetype},
                        "absolute": {u_high,s_high,archetype}}}.
    """
    util = {d: mean_marginal.get(d, {}).get('contribution', float('nan')) for d in algorithm_list}
    stab_mean, stab_trend = {}, {}
    for d in algorithm_list:
        ys = survival.get(d, [])
        stab_mean[d] = float(np.mean(ys)) if ys else float('nan')
        stab_trend[d] = (ys[-1] - ys[0]) if ys else float('nan')

    med_u = _finite_median(list(util.values()))
    med_s = _finite_median(list(stab_mean.values()))

    out: Dict[str, Dict[str, Any]] = {}
    for d in algorithm_list:
        u, sm, st = util[d], stab_mean[d], stab_trend[d]
        util_nan = bool(np.isnan(u))

        schemes: Dict[str, Dict[str, Any]] = {}
        for scheme, (tu, ts) in (
            ("relative", (med_u, med_s)),
            ("absolute", (abs_utility, abs_stability)),
        ):
            u_high = (not np.isnan(u)) and (not np.isnan(tu)) and (u > tu)
            # Stability depends only on the mean survival rate (no trend gate).
            s_high = (not np.isnan(sm)) and (not np.isnan(ts)) and (sm > ts)
            schemes[scheme] = {
                "u_high": u_high, "s_high": s_high,
                "archetype": _assign_archetype(u_high, s_high, util_nan),
            }

        out[d] = {
            "utility": u,
            "stability_mean": sm, "stability_trend": st,
            "relative": schemes["relative"], "absolute": schemes["absolute"],
        }
    return out


# ── Plot helpers ────────────────────────────────────────────────────────────

def _ga_plot_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def plot_ga_utility(
    lofo: Dict[str, float],
    mean_marginal: Dict[str, Dict[str, float]],
    best_ensemble: List[str],
    algorithm_list: List[str],
    dataset: str,
    entity: str,
) -> None:
    """
    Two-panel bar chart for Axis 1.
      Top    — LOFO marginal for each detector in best_ensemble.
               Bars are coloured by sign: green = removal hurts fitness (the
               detector is pulling weight); red = removal helps (its removal
               would actually improve fitness).
      Bottom — mean marginal contribution across evaluated subsets for ALL
               detectors. NaN values are drawn as faded grey bars at zero.

    Saves to myresults/GA_Ens/{dataset}/{entity}/ga_selection_utility_{dataset}_{entity}.png.
    """
    _ga_plot_rcparams()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(8, 0.55 * len(algorithm_list) + 4), 7),
        gridspec_kw={"height_ratios": [1, 1]},
    )

    # Top: LOFO on best_ensemble
    if best_ensemble:
        vals = [lofo.get(d, float('nan')) for d in best_ensemble]
        colours = []
        for v in vals:
            if np.isnan(v):
                colours.append("#888888")
            elif v >= 0:
                colours.append("#2ca02c")
            else:
                colours.append("#d62728")
        x = np.arange(len(best_ensemble))
        ax_top.bar(x, [0.0 if np.isnan(v) else v for v in vals], color=colours)
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(best_ensemble, rotation=30, ha="right")
        ax_top.axhline(0, color="black", linewidth=0.6)
        ax_top.set_ylabel("fitness(best) − fitness(best \\ detector)")
        ax_top.set_title(
            f"Axis 1a · LOFO on best ensemble: {best_ensemble}")
        ax_top.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    else:
        ax_top.text(0.5, 0.5, "best_ensemble is empty",
                    ha="center", va="center", transform=ax_top.transAxes)

    # Bottom: mean marginal contribution for ALL detectors
    raw = [mean_marginal.get(d, {}).get('contribution', float('nan'))
           for d in algorithm_list]
    bot_vals = [0.0 if np.isnan(v) else v for v in raw]
    bot_colours = ["#cccccc" if np.isnan(v) else
                   ("#2ca02c" if v >= 0 else "#d62728") for v in raw]
    xb = np.arange(len(algorithm_list))
    ax_bot.bar(xb, bot_vals, color=bot_colours)
    ax_bot.set_xticks(xb)
    ax_bot.set_xticklabels(algorithm_list, rotation=30, ha="right")
    ax_bot.axhline(0, color="black", linewidth=0.6)
    ax_bot.set_ylabel("E[fit | present] − E[fit | absent]")
    ax_bot.set_title(
        "Axis 1b · Mean marginal contribution across evaluated subsets")
    ax_bot.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout(pad=1.2)
    directory = f"myresults/GA_Ens/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/ga_selection_utility_{dataset}_{entity}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_ga_interaction(
    friedman_h: Dict[str, Any],
    algorithm_list: List[str],
    dataset: str,
    entity: str,
) -> None:
    """
    Heatmap of the two-way Friedman H-statistic H_jk (Eq. 44). H ≥ 0, so a
    sequential colormap is used (0 = additive / no interaction, higher = stronger
    interaction). NaN cells (undefined) are drawn grey; diagonal is masked.

    Saves to myresults/GA_Ens/{dataset}/{entity}/ga_selection_interaction_{dataset}_{entity}.png.
    """
    _ga_plot_rcparams()
    H_two = friedman_h.get("H_two_way", {})
    n = len(algorithm_list)
    # Keep only the upper-right triangle (row < col); the matrix is symmetric so
    # the lower triangle is the same values mirrored and is left blank.
    M = np.full((n, n), np.nan, dtype=float)
    for (j, k), v in H_two.items():
        if j == k:
            continue
        try:
            r, c = algorithm_list.index(j), algorithm_list.index(k)
        except ValueError:
            continue
        if r < c:
            M[r, c] = v

    fig, ax = plt.subplots(figsize=(max(7, 0.6 * n + 2), max(6, 0.55 * n + 2)))
    masked = np.ma.array(M, mask=np.isnan(M))
    finite = M[np.isfinite(M)]
    vmax = float(np.nanmax(finite)) if finite.size > 0 and np.nanmax(finite) > 0 else 1.0
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(algorithm_list, rotation=45, ha="right")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(algorithm_list)

    for jj in range(n):
        for kk in range(n):
            v = M[jj, kk]
            if jj == kk or np.isnan(v):
                continue
            shade = "white" if v < 0.5 * vmax else "black"
            ax.text(kk, jj, f"{v:.3f}", ha="center", va="center",
                    color=shade, fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label="H_jk (two-way)")
    r2 = friedman_h.get("surrogate_r2", float('nan'))
    sub = (f"surrogate={friedman_h.get('surrogate', '?')}, "
           f"R²={r2:.3f}" if not np.isnan(r2) else
           f"surrogate={friedman_h.get('surrogate', '?')}")
    ax.set_title("Axis 2 · Friedman two-way interaction  H_jk (Eq. 44)\n"
                 f"({sub}, n_subsets={friedman_h.get('n_subsets', 0)})")
    plt.tight_layout(pad=1.2)

    directory = f"myresults/GA_Ens/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/ga_selection_interaction_{dataset}_{entity}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_ga_total_interaction(
    friedman_h: Dict[str, Any],
    algorithm_list: List[str],
    best_ensemble: List[str],
    dataset: str,
    entity: str,
) -> None:
    """
    Bar chart of the total Friedman H-statistic H_j (Eq. 45) — each detector's
    overall interaction strength with all others. Members of best_ensemble are
    drawn opaque; the rest faded. NaN values are drawn as faded grey zero bars.

    Saves to myresults/GA_Ens/{dataset}/{entity}/ga_selection_total_interaction_{dataset}_{entity}.png.
    """
    _ga_plot_rcparams()
    H_tot = friedman_h.get("H_total", {})
    in_best = set(best_ensemble or [])
    raw = [H_tot.get(d, float('nan')) for d in algorithm_list]
    vals = [0.0 if np.isnan(v) else v for v in raw]
    colours = []
    for d, v in zip(algorithm_list, raw):
        if np.isnan(v):
            colours.append("#cccccc")
        elif d in in_best:
            colours.append("#1f77b4")
        else:
            colours.append("#9ecae1")

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(algorithm_list) + 4), 5))
    x = np.arange(len(algorithm_list))
    ax.bar(x, vals, color=colours)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithm_list, rotation=30, ha="right")
    ax.set_ylabel("H_j  (total interaction)")
    ax.set_ylim(0, max(1.0, max(vals) * 1.1) if vals else 1.0)
    ax.set_title("Axis 2 · Friedman total interaction  H_j (Eq. 45)  "
                 "— dark = best-ensemble members")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout(pad=1.2)
    directory = f"myresults/GA_Ens/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/ga_selection_total_interaction_{dataset}_{entity}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_ga_survival_impl(
    survival_rates: Dict[str, List[float]],
    bold_set: set,
    title: str,
    save_path: str,
) -> None:
    """Shared rendering for survival-rate plots. `bold_set` controls which
    detectors receive the bold/opaque/marker treatment."""
    _ga_plot_rcparams()
    detectors = list(survival_rates.keys())
    if not detectors:
        return
    G = len(next(iter(survival_rates.values())))
    colour_map = {d: plt.cm.tab20(i / max(len(detectors), 1)) for i, d in enumerate(detectors)}

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * G + 4), 5))
    x = np.arange(G)
    for d in detectors:
        ys = survival_rates[d]
        bold = d in bold_set
        ax.plot(x, ys, label=d, color=colour_map[d],
                linewidth=2.0 if bold else 1.0,
                alpha=1.0 if bold else 0.35,
                marker="o" if bold else None,
                markersize=3.5 if bold else 0)

    ax.set_xlabel("Generation")
    ax.set_ylabel("P(d, g)  —  survival rate")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", ncol=1, frameon=False,
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

    plt.tight_layout(pad=1.2)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_ga_survival(
    survival_rates: Dict[str, List[float]],
    best_ensemble: List[str],
    dataset: str,
    entity: str,
) -> None:
    """
    Produce two survival-rate line plots for Axis 3.

    (a) Ensemble-highlighted version — detectors in best_ensemble are bold/opaque;
        all others are faded. Emphasises which detectors the GA converged on.
        → ga_selection_survival_{dataset}_{entity}.png

    (b) All-bold version — every detector is drawn equally bold and opaque,
        so their trajectories can be compared without any pre-selection bias.
        → ga_selection_survival_all_{dataset}_{entity}.png
    """
    directory = f"myresults/GA_Ens/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    in_best = set(best_ensemble or [])

    _plot_ga_survival_impl(
        survival_rates,
        bold_set=in_best,
        title=("Axis 3 · Evolutionary survival per detector "
               "(bold = members of best ensemble)"),
        save_path=f"{directory}/ga_selection_survival_{dataset}_{entity}.png",
    )
    _plot_ga_survival_impl(
        survival_rates,
        bold_set=set(survival_rates.keys()),   # every detector bold
        title="Axis 3 · Evolutionary survival per detector (all detectors)",
        save_path=f"{directory}/ga_selection_survival_all_{dataset}_{entity}.png",
    )


def plot_ga_archetypes(
    archetypes: Dict[str, Dict[str, Any]],
    algorithm_list: List[str],
    dataset: str,
    entity: str,
) -> None:
    """
    Two-panel scatter of the axis intersection that drives the archetypes.
    Left = relative (median-split) scheme, right = absolute (fixed cutoffs).

      x = utility (mean marginal contribution), y = stability (mean survival rate)
      colour       = assigned archetype (shared categorical palette)
      dashed lines = that scheme's utility / stability thresholds

    (Complementarity / Friedman H axis is currently disabled.) Unclassified
    detectors (NaN utility) are not plotted; they are listed in a caption.
    Saves to ga_selection_archetypes_{dataset}_{entity}.png.
    """
    _ga_plot_rcparams()
    # Shared archetype → colour palette.
    palette = {name: plt.cm.tab10(i / max(len(ARCHETYPE_ORDER), 1))
               for i, name in enumerate(ARCHETYPE_ORDER)}

    util = {d: archetypes[d]["utility"] for d in algorithm_list}
    stab = {d: archetypes[d]["stability_mean"] for d in algorithm_list}
    med_u = _finite_median(list(util.values()))
    med_s = _finite_median(list(stab.values()))
    thresholds = {
        "relative": (med_u, med_s),
        "absolute": (0.0, 0.5),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    unclassified = sorted({d for d in algorithm_list
                           if archetypes[d]["relative"]["archetype"] == ARCHETYPE_UNCLASSIFIED})
    seen_labels = set()

    for ax, scheme in zip(axes, ("relative", "absolute")):
        tu, ts = thresholds[scheme]
        for d in algorithm_list:
            u, s = util[d], stab[d]
            info = archetypes[d][scheme]
            if np.isnan(u) or np.isnan(s):
                continue   # cannot place a point without both coordinates
            arche = info["archetype"]
            colour = palette.get(arche, "#888888")
            label = arche if arche not in seen_labels else None
            seen_labels.add(arche)
            ax.scatter([u], [s], s=90, color=colour,
                       edgecolors=colour, linewidths=1.5,
                       facecolors=colour, label=label, zorder=3)
            ax.annotate(d, (u, s), textcoords="offset points", xytext=(5, 4),
                        fontsize=8, alpha=0.85)
        if not np.isnan(tu):
            ax.axvline(tu, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        if not np.isnan(ts):
            ax.axhline(ts, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Utility  (mean marginal contribution)")
        ax.set_ylabel("Stability  (mean survival rate)")
        ax.set_title(f"{scheme.capitalize()} thresholds")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.6),
               frameon=False, title="Archetype")
    if unclassified:
        fig.text(0.5, -0.02, "Unclassified (no marginal-contribution data): "
                 + ", ".join(unclassified), ha="center", fontsize=9, alpha=0.8)

    fig.suptitle("Functional Archetypes · utility × stability",
                 y=1.02)
    plt.tight_layout(pad=1.2)
    directory = f"myresults/GA_Ens/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/ga_selection_archetypes_{dataset}_{entity}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


# ── Orchestrator + report ───────────────────────────────────────────────────

def explain_ga_selection(
    best_ensemble: List[str],
    evaluated_ensembles: Dict[Tuple[str, ...], tuple],
    generation_populations: List[List[List[str]]],
    algorithm_list: List[str],
    population_size: int,
    evaluate_fitness: Callable[[List[str]], float],
    dataset: str,
    entity: str,
    explain: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    GA-ensemble selection explainability: explain *why* each detector ended up
    in best_ensemble, along two analytical axes (utility, stability). The
    Complementarity / Friedman H axis is currently disabled. Produces three plots
    and a structured text report under
        myresults/GA_Ens/{dataset}/{entity}/

    Returns a dict with the computed structures when explain=True; None otherwise.
    """
    if not explain:
        return None
    if not best_ensemble:
        return None

    lofo = compute_lofo_utility(best_ensemble, evaluate_fitness)
    mean_marginal = compute_mean_marginal_contribution(evaluated_ensembles, algorithm_list)
    # --- Complementarity axis (Friedman H) temporarily disabled ---
    # friedman_h = compute_friedman_h(evaluated_ensembles, algorithm_list)
    # interaction = friedman_h["H_two_way"]
    survival = compute_survival_rates(generation_populations, algorithm_list, population_size)
    archetypes = classify_detector_archetypes(
        mean_marginal, survival, algorithm_list)

    plot_ga_utility(lofo, mean_marginal, best_ensemble, algorithm_list, dataset, entity)
    # plot_ga_interaction(friedman_h, algorithm_list, dataset, entity)
    # plot_ga_total_interaction(friedman_h, algorithm_list, best_ensemble, dataset, entity)
    plot_ga_survival(survival, best_ensemble, dataset, entity)
    plot_ga_archetypes(archetypes, algorithm_list, dataset, entity)

    directory = f"myresults/GA_Ens/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    report_path = os.path.join(
        directory, f"ga_selection_explainability_{dataset}_{entity}.txt")

    n_subsets = len(evaluated_ensembles)
    n_generations = len(generation_populations)

    with open(report_path, "w") as f:
        f.write("=== GA Ensemble Selection Explainability ===\n")
        f.write(f"Dataset: {dataset}  |  Entity: {entity}\n")
        f.write(f"Best ensemble  : {list(best_ensemble)}\n")
        f.write(f"Population size: {population_size}\n")
        f.write(f"Generations    : {n_generations}\n")
        f.write(f"Distinct subsets evaluated: {n_subsets}\n\n")

        # ── Axis 1 ────────────────────────────────────────────────────────
        backslash_error_text = r"fitness(best) - fitness(best \ d)"
        f.write("--- Axis 1: Utility ---\n")
        f.write("(1a) LOFO marginal fitness change (final ensemble):\n")
        f.write(f"      {'detector':<14} {backslash_error_text:>40}\n")
        f.write("      " + "-" * 56 + "\n")
        for d in best_ensemble:
            v = lofo.get(d, float('nan'))
            s = f"{v:+.4f}" if not np.isnan(v) else "N/A"
            f.write(f"      {d:<14} {s:>40}\n")

        f.write("\n(1b) Mean marginal contribution across all evaluated subsets:\n")
        f.write(f"      {'detector':<14} {'E[fit|p]-E[fit|a]':>18} "
                f"{'E[fit|p]':>10} {'E[fit|a]':>10} {'#p':>5} {'#a':>5}\n")
        f.write("      " + "-" * 68 + "\n")
        for d in algorithm_list:
            mm = mean_marginal[d]
            c = f"{mm['contribution']:+.4f}" if not np.isnan(mm['contribution']) else "N/A"
            ep = f"{mm['e_present']:.4f}" if not np.isnan(mm['e_present']) else "N/A"
            ea = f"{mm['e_absent']:.4f}" if not np.isnan(mm['e_absent']) else "N/A"
            f.write(f"      {d:<14} {c:>18} {ep:>10} {ea:>10} "
                    f"{mm['n_present']:>5d} {mm['n_absent']:>5d}\n")

        # ── Axis 2 (Complementarity / Friedman H) — temporarily disabled ───
        f.write("\n--- Axis 2: Inter-model Complementarity (Friedman H-statistic) ---\n")
        f.write("(Temporarily disabled — the Complementarity axis is currently left out.)\n")

        # ── Axis 3 ────────────────────────────────────────────────────────
        f.write("\n--- Axis 3: Stability (Evolutionary Survival) ---\n")
        f.write("P(d, g) = (#individuals in generation g containing d) / population_size\n")
        f.write(f"      {'detector':<14} {'mean P':>8} {'first':>8} "
                f"{'last':>8} {'trend':>10}\n")
        f.write("      " + "-" * 52 + "\n")
        for d in algorithm_list:
            ys = survival[d]
            if not ys:
                f.write(f"      {d:<14} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>10}\n")
                continue
            f.write(f"      {d:<14} {np.mean(ys):>8.3f} {ys[0]:>8.3f} "
                    f"{ys[-1]:>8.3f} {(ys[-1] - ys[0]):>+10.3f}\n")

        # ── Synthesis ──────────────────────────────────────────────────────
        # (Complementarity columns total-H_j / mean-H-with-peers disabled.)
        f.write("\n--- Synthesis: why each detector of the best ensemble was selected ---\n")
        f.write(f"      {'detector':<14} {'LOFO':>10} {'mean marg.':>12} "
                f"{'last-gen P':>12}\n")
        f.write("      " + "-" * 50 + "\n")
        for d in best_ensemble:
            lv = lofo.get(d, float('nan'))
            mv = mean_marginal[d]['contribution']
            last_p = survival[d][-1] if survival[d] else float('nan')

            def _sgn(v):
                return f"{v:+.4f}" if not np.isnan(v) else "N/A"

            f.write(
                f"      {d:<14} {_sgn(lv):>10} {_sgn(mv):>12} "
                f"{(f'{last_p:.3f}' if not np.isnan(last_p) else 'N/A'):>12}\n"
            )

        # ── Functional archetypes ──────────────────────────────────────────
        # (Complementarity axis disabled — archetypes use Utility × Stability.)
        f.write("\n--- Functional Archetypes (axis intersections) ---\n")
        f.write("Utility = mean marginal contribution (Axis 1b only; LOFO excluded).\n")
        f.write("Stability = mean survival rate "
                "(trend P_last − P_first is shown for context but does not affect classification).\n")
        f.write("Two threshold schemes reported: relative (median split) | absolute "
                "(util>0, surv>0.5).\n")
        f.write("Archetype = the (U,S) high/low pair as a 2-letter code, e.g. "
                "HL = high utility, low stability.\n\n")
        f.write(f"      {'detector':<14} {'util':>9} {'stab':>7} "
                f"{'trend':>8}  {'archetype[rel]':<16} {'archetype[abs]'}\n")
        f.write("      " + "-" * 78 + "\n")

        def _num(v, fmt="{:+.4f}"):
            return fmt.format(v) if not np.isnan(v) else "N/A"

        for d in algorithm_list:
            a = archetypes[d]
            f.write(
                f"      {d:<14} {_num(a['utility']):>9} "
                f"{_num(a['stability_mean'], '{:.3f}'):>7} {_num(a['stability_trend'], '{:+.3f}'):>8}  "
                f"{a['relative']['archetype']:<16} {a['absolute']['archetype']}\n"
            )

        for scheme in ("relative", "absolute"):
            tally: Dict[str, int] = {}
            for d in algorithm_list:
                name = archetypes[d][scheme]["archetype"]
                tally[name] = tally.get(name, 0) + 1
            ordered = [(nm, tally[nm]) for nm in ARCHETYPE_ORDER if nm in tally]
            summary = ", ".join(f"{nm}: {ct}" for nm, ct in ordered)
            f.write(f"\n  Tally [{scheme}]: {summary}\n")

    result = {
        "best_ensemble": list(best_ensemble),
        "lofo": lofo,
        "mean_marginal": mean_marginal,
        # Complementarity axis disabled: friedman_h / H_two_way / H_total omitted.
        "survival": survival,
        "archetypes": archetypes,
        "n_subsets_evaluated": n_subsets,
        "n_generations": n_generations,
    }

    # ── Intermediate Representation (grounded LLM input; non-fatal) ─────────
    try:
        _ir = _ir_module()
        _ir.write_stage_ir(_ir.build_ga_selection_ir(dataset, entity, result),
                           dataset, entity, "ir_ga_selection")
    except Exception as e:
        logger.error(f"GA selection IR emission failed (non-fatal): {e}")

    return result


# ════════════════════════════════════════════════════════════════════════════
#  GA Ensemble Combination Explainability
#
#  Explains *how the meta-learner combines* the chosen detectors, by attributing
#  its output to the per-detector score columns via two methods, then merging
#  their rankings with a Markov-chain rank aggregation:
#    • SHAP — exact interventional Shapley (single mean baseline), label-free.
#    • PFI  — permutation feature importance measured as F1 drop, label-based.
# ════════════════════════════════════════════════════════════════════════════


def _best_threshold_f1(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Maximum F1 over the same threshold grid fitness_function uses."""
    y_true = np.asarray(y_true).flatten()
    y_scores = np.asarray(y_scores).flatten()
    best = 0.0
    for t in np.linspace(0.1, 0.9, 50):
        y_pred = (y_scores >= t).astype(int)
        f1 = f1_score(y_pred, y_true)[0]
        if f1 > best:
            best = f1
    return float(best)


def compute_meta_shap(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_explain: np.ndarray,
    baseline_row: np.ndarray,
    feature_names: List[str],
    mode: str = "abs",
) -> Dict[str, float]:
    """
    Exact interventional Shapley values of the meta-learner over its detector
    features, using a SINGLE mean baseline. Global importance per feature:
      mode="abs"    → mean over explained rows of |phi_i(x)|  (magnitude of influence)
      mode="signed" → mean over explained rows of  phi_i(x)   (net direction of influence)

    For instance x and subset S of features, F(S) marginalises the absent
    features to the baseline:  z_j = x_j if j in S else baseline_row[j].
    phi_i(x) = Σ_{S ⊆ F\\{i}} w(|S|) (F(S∪i) − F(S)),  w(s)=s!(d−s−1)!/d!.

    Because features are binary-ish score columns and d = ensemble size is small,
    all 2^d subset predictions are enumerated exactly (cheap).
    """
    _agg = (lambda p: float(np.mean(np.abs(p)))) if mode == "abs" else (lambda p: float(np.mean(p)))
    d = len(feature_names)
    n = X_explain.shape[0]
    if d == 0 or n == 0:
        return {f: float('nan') for f in feature_names}
    if d == 1:
        # Single feature carries the entire deviation from baseline.
        Z1 = X_explain.copy()
        Z0 = np.tile(baseline_row, (n, 1))
        phi = np.asarray(predict_fn(Z1), float) - np.asarray(predict_fn(Z0), float)
        return {feature_names[0]: _agg(phi)}

    import math
    # Cache F(S) for every subset mask (bitmask over feature indices).
    pred_cache: Dict[int, np.ndarray] = {}
    for mask in range(1 << d):
        Z = np.tile(baseline_row.astype(float), (n, 1))
        for j in range(d):
            if mask & (1 << j):
                Z[:, j] = X_explain[:, j]
        pred_cache[mask] = np.nan_to_num(np.asarray(predict_fn(Z), dtype=float),
                                         nan=0.0, posinf=1.0, neginf=0.0)

    fact = math.factorial
    weight = {s: fact(s) * fact(d - s - 1) / fact(d) for s in range(d)}
    out: Dict[str, float] = {}
    for i in range(d):
        phi = np.zeros(n, dtype=float)
        others = [j for j in range(d) if j != i]
        # Enumerate every subset S of the other features.
        for sub in range(1 << (d - 1)):
            mask = 0
            s = 0
            for b, j in enumerate(others):
                if sub & (1 << b):
                    mask |= (1 << j)
                    s += 1
            phi += weight[s] * (pred_cache[mask | (1 << i)] - pred_cache[mask])
        out[feature_names[i]] = _agg(phi)
    return out


def compute_meta_pfi(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    score_fn: Callable[[np.ndarray, np.ndarray], float] = _best_threshold_f1,
    n_repeats: int = 10,
    random_state: int = 0,
) -> Dict[str, float]:
    """
    Permutation feature importance of the meta-learner: importance of feature i =
    baseline_score − mean over n_repeats of the score after shuffling column i.
    score_fn defaults to best-threshold F1 (matches GA fitness); injectable.
    """
    d = len(feature_names)
    n = X.shape[0]
    if d == 0 or n == 0:
        return {f: float('nan') for f in feature_names}
    y = np.asarray(y).flatten()
    base_scores = np.nan_to_num(np.asarray(predict_fn(X), float),
                                nan=0.0, posinf=1.0, neginf=0.0)
    baseline = score_fn(y, base_scores)
    rng = np.random.RandomState(random_state)
    out: Dict[str, float] = {}
    for i in range(d):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, i] = Xp[rng.permutation(n), i]
            sp = np.nan_to_num(np.asarray(predict_fn(Xp), float),
                               nan=0.0, posinf=1.0, neginf=0.0)
            drops.append(baseline - score_fn(y, sp))
        out[feature_names[i]] = float(np.mean(drops))
    return out


def markov_aggregate_importances(
    importances_by_method: Dict[str, Dict[str, float]],
    feature_names: List[str],
    smoothing: float = 0.1,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Markov-chain rank aggregation over each method's importance ranking — the same
    family as enhanced_markov_chain_rank_aggregator_text in
    Model_Selection/rank_aggregation.py, reimplemented here (numpy-only) so the
    Metrics explainability layer stays free of that module's heavy deps and exposes
    a per-feature stationary score.

    Each method induces a best-first ranking (descending importance; NaN last). A
    pairwise preference matrix C[i,j] counts the methods that rank i above j. The
    chain moves preferentially toward the better item: P[i,j] ∝ sigmoid(C[j,i] −
    C[i,j]) for i≠j, with a small Laplace `smoothing` added to every off-diagonal so
    the chain is irreducible/aperiodic (a unique stationary distribution exists);
    rows are then normalised. The stationary distribution π (left eigenvector of P
    for eigenvalue 1) is the per-feature Markov score — higher π = stronger consensus
    importance. Returns ({feature: π}, final_ranking_descending).
    """
    n = len(feature_names)
    if n == 0:
        return {}, []
    if n == 1:
        return {feature_names[0]: 1.0}, list(feature_names)

    idx = {f: i for i, f in enumerate(feature_names)}

    # Pairwise preference counts from each method's best-first ranking.
    C = np.zeros((n, n), dtype=float)
    for imp in importances_by_method.values():
        order = sorted(
            feature_names,
            key=lambda f: (imp.get(f, float('nan'))
                           if not np.isnan(imp.get(f, float('nan'))) else -np.inf),
            reverse=True,
        )
        for a in range(n):
            for b in range(a + 1, n):
                C[idx[order[a]], idx[order[b]]] += 1.0  # order[a] preferred over order[b]

    # Transition matrix: move toward the better item; Laplace smoothing → ergodic.
    P = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                P[i, j] = 1.0 / (1.0 + np.exp(-(C[j, i] - C[i, j]))) + smoothing
    P /= P.sum(axis=1, keepdims=True)

    # Stationary distribution: left eigenvector of P for the eigenvalue closest to 1.
    vals, vecs = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.abs(vecs[:, k].real)
    total = pi.sum()
    pi = pi / total if total > 0 else np.full(n, 1.0 / n)

    scores = {f: float(pi[idx[f]]) for f in feature_names}
    final_ranking = sorted(feature_names, key=lambda f: scores[f], reverse=True)
    return scores, final_ranking


def _competition_ranks(points: Dict[str, float], order: List[str]) -> Dict[str, int]:
    """
    Standard competition ranking ("1224"): features with equal points share the
    smallest rank in their group (so two features tied for 2nd are both rank 2 and
    the next is rank 4). `order` must be the points-descending feature order.
    """
    ranks: Dict[str, int] = {}
    prev_pts = None
    rank = 0
    for i, f in enumerate(order):
        if prev_pts is None or points[f] != prev_pts:
            rank = i + 1
            prev_pts = points[f]
        ranks[f] = rank
    return ranks


def plot_ga_combination(
    shap_abs: Dict[str, float],
    shap_signed: Dict[str, float],
    pfi_imp: Dict[str, float],
    markov_scores: Dict[str, float],
    final_ranking: List[str],
    feature_names: List[str],
    dataset: str,
    entity: str,
) -> None:
    """
    Two-panel summary of the meta-learner weighting.
      Left  — grouped horizontal bars per detector: mean|SHAP|, mean SHAP (signed),
              and PFI importances, each normalised to its own max abs so the three
              methods are comparable (signed SHAP can be negative).
      Right — Markov final ranking: horizontal bars of the stationary-probability
              score, winner on top.

    Saves to ga_combination_importance_{dataset}_{entity}.png.
    """
    _ga_plot_rcparams()

    def _norm(dct):
        vals = [dct.get(f, float('nan')) for f in feature_names]
        m = np.nanmax(np.abs(vals)) if np.any(~np.isnan(vals)) else 0.0
        m = m if m > 0 else 1.0
        return [0.0 if np.isnan(v) else v / m for v in vals]

    fig, (ax_imp, ax_markov) = plt.subplots(1, 2, figsize=(13, max(4, 0.6 * len(feature_names) + 2)))

    y = np.arange(len(feature_names))
    h = 0.27
    ax_imp.barh(y - h, _norm(shap_abs), height=h, label="mean|SHAP|", color="#1f77b4")
    ax_imp.barh(y, _norm(shap_signed), height=h, label="mean SHAP (signed)", color="#9467bd")
    ax_imp.barh(y + h, _norm(pfi_imp), height=h, label="PFI (F1 drop)", color="#ff7f0e")
    ax_imp.axvline(0, color="black", linewidth=0.6)
    ax_imp.set_yticks(y)
    ax_imp.set_yticklabels(feature_names)
    ax_imp.invert_yaxis()
    ax_imp.set_xlabel("Importance (normalised to each method's max |·|)")
    ax_imp.set_title("Meta-learner feature attribution")
    ax_imp.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    ax_imp.legend(loc="lower right", frameon=False)

    ranked = list(final_ranking)
    pts = [markov_scores.get(f, 0.0) for f in ranked]
    yb = np.arange(len(ranked))
    # Competition ranks so tied Markov scores share a rank number on the labels.
    rk = _competition_ranks(markov_scores, ranked)
    ax_markov.barh(yb, pts, color="#2ca02c")
    ax_markov.set_yticks(yb)
    ax_markov.set_yticklabels([f"{rk[f]}. {f}" for f in ranked])
    ax_markov.invert_yaxis()
    ax_markov.set_xlabel("Markov score (stationary prob.)")
    ax_markov.set_title("Final ranking (Markov: SHAP + PFI)")
    ax_markov.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout(pad=1.2)
    directory = f"myresults/GA_Ens/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/ga_combination_importance_{dataset}_{entity}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


def explain_ga_combination(
    best_ensemble: List[str],
    algorithm_list: List[str],
    base_model_predictions_train: np.ndarray,
    base_model_predictions_test: np.ndarray,
    y_true_train: np.ndarray,
    y_true_test: np.ndarray,
    meta_model_type: str,
    dataset: str,
    entity: str,
    meta_model: Any = None,
    predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    max_explain: int = 200,
    explain: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Combination-layer explainability: attribute the best-ensemble meta-learner's
    output to its detector score-columns via SHAP and PFI, then merge the three
    rankings (mean|SHAP|, signed SHAP, PFI) with a Markov-chain rank aggregation.
    Writes a report + plot under myresults/GA_Ens/{dataset}/{entity}/ and returns a
    dict (None if explain=False).

    The meta-learner is, in priority order: an injected predict_fn (tests); the
    GA's actual captured meta_model; or — only as a defensive fallback — a freshly
    trained model of meta_model_type. The captured model is validated against the
    feature count and the fallback is used on any mismatch.
    """
    if not explain or not best_ensemble:
        return None

    feature_names = [a for a in algorithm_list if a in best_ensemble]
    d = len(feature_names)
    if d == 0:
        return None

    header = np.array(algorithm_list)
    mask = np.isin(header, best_ensemble)
    X_train_f = np.asarray(base_model_predictions_train, dtype=float)[:, mask]
    X_test_f = np.asarray(base_model_predictions_test, dtype=float)[:, mask]
    X_train_f = np.nan_to_num(X_train_f, nan=0.0, posinf=1.0, neginf=0.0)
    X_test_f = np.nan_to_num(X_test_f, nan=0.0, posinf=1.0, neginf=0.0)

    used_source = "injected"
    if predict_fn is None:
        candidate = meta_model
        # Validate the captured model's expected feature width when discoverable.
        ok = candidate is not None
        n_in = getattr(candidate, "n_features_in_", None)
        if ok and n_in is not None and n_in != d:
            ok = False
        if ok:
            predict_fn = lambda Z: candidate.predict_proba(Z)[:, 1]
            used_source = "captured"
        else:
            # Defensive fallback: retrain a meta-model of the requested type.
            yt = np.asarray(y_true_train).flatten()
            if len(np.unique(yt)) < 2:
                yt = inject_synthetic_anomalies(yt)
            trainer = {
                'lr': train_meta_model, 'rf': train_meta_model_rf,
                'gbm': train_meta_model_gbm, 'svm': train_meta_model_svm,
            }.get(meta_model_type, train_meta_model_rf)
            model = trainer(X_train_f, yt)
            predict_fn = lambda Z: model.predict_proba(Z)[:, 1]
            used_source = "retrained_fallback"

    # Subsample explained rows for SHAP speed (deterministic).
    n_test = X_test_f.shape[0]
    if n_test > max_explain:
        rng = np.random.RandomState(0)
        idx = rng.choice(n_test, size=max_explain, replace=False)
        X_explain = X_test_f[idx]
    else:
        X_explain = X_test_f
    baseline_row = X_train_f.mean(axis=0) if X_train_f.shape[0] > 0 else np.zeros(d)

    shap_abs = compute_meta_shap(predict_fn, X_explain, baseline_row, feature_names, mode="abs")
    shap_signed = compute_meta_shap(predict_fn, X_explain, baseline_row, feature_names, mode="signed")
    pfi_imp = compute_meta_pfi(predict_fn, X_test_f, y_true_test, feature_names)
    markov_scores, final_ranking = markov_aggregate_importances(
        {"SHAP_abs": shap_abs, "SHAP_signed": shap_signed, "PFI": pfi_imp}, feature_names)

    baseline_f1 = _best_threshold_f1(
        y_true_test, np.nan_to_num(np.asarray(predict_fn(X_test_f), float),
                                   nan=0.0, posinf=1.0, neginf=0.0))

    plot_ga_combination(shap_abs, shap_signed, pfi_imp, markov_scores, final_ranking,
                        feature_names, dataset, entity)

    # Per-method ranks (1 = most important / most positive).
    def _ranks(imp):
        order = sorted(feature_names,
                       key=lambda f: (imp[f] if not np.isnan(imp[f]) else -np.inf),
                       reverse=True)
        return {f: i + 1 for i, f in enumerate(order)}
    shap_abs_rank, shap_signed_rank, pfi_rank = _ranks(shap_abs), _ranks(shap_signed), _ranks(pfi_imp)

    directory = f"myresults/GA_Ens/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    report_path = os.path.join(
        directory, f"ga_combination_explainability_{dataset}_{entity}.txt")
    with open(report_path, "w") as f:
        f.write("=== GA Ensemble Combination Explainability ===\n")
        f.write(f"Dataset: {dataset}  |  Entity: {entity}\n")
        f.write(f"Best ensemble : {list(best_ensemble)}\n")
        f.write(f"Meta-learner  : {meta_model_type}  (model source: {used_source})\n")
        f.write(f"Features (detector score columns): {d}\n")
        f.write(f"Baseline meta-learner F1 (best threshold): {baseline_f1:.4f}\n\n")

        f.write("--- SHAP |.| (mean |SHAP|: magnitude of contribution; label-free) ---\n")
        f.write(f"      {'detector':<14} {'mean|SHAP|':>12} {'rank':>6}\n")
        f.write("      " + "-" * 34 + "\n")
        for f_ in sorted(feature_names, key=lambda x: shap_abs_rank[x]):
            v = shap_abs[f_]
            vs = f"{v:.6f}" if not np.isnan(v) else "N/A"
            f.write(f"      {f_:<14} {vs:>12} {shap_abs_rank[f_]:>6}\n")

        f.write("\n--- SHAP signed (mean SHAP: net direction of contribution; label-free) ---\n")
        f.write(f"      {'detector':<14} {'mean SHAP':>12} {'rank':>6}\n")
        f.write("      " + "-" * 34 + "\n")
        for f_ in sorted(feature_names, key=lambda x: shap_signed_rank[x]):
            v = shap_signed[f_]
            vs = f"{v:+.6f}" if not np.isnan(v) else "N/A"
            f.write(f"      {f_:<14} {vs:>12} {shap_signed_rank[f_]:>6}\n")

        f.write("\n--- PFI (F1 drop when the detector's column is shuffled; label-based) ---\n")
        f.write(f"      {'detector':<14} {'F1 drop':>12} {'rank':>6}\n")
        f.write("      " + "-" * 34 + "\n")
        for f_ in sorted(feature_names, key=lambda x: pfi_rank[x]):
            v = pfi_imp[f_]
            vs = f"{v:+.6f}" if not np.isnan(v) else "N/A"
            f.write(f"      {f_:<14} {vs:>12} {pfi_rank[f_]:>6}\n")

        f.write("\n--- Markov aggregation (SHAP |.| + SHAP signed + PFI) ---\n")
        f.write(f"      {'detector':<14} {'|SHAP| rk':>9} {'SHAP rk':>8} "
                f"{'PFI rk':>7} {'Markov π':>10}\n")
        f.write("      " + "-" * 52 + "\n")
        for f_ in final_ranking:
            f.write(f"      {f_:<14} {shap_abs_rank[f_]:>9} {shap_signed_rank[f_]:>8} "
                    f"{pfi_rank[f_]:>7} {markov_scores[f_]:>10.4f}\n")
        # Final ranking with ties shown as equals (e.g. "1.A > 2.B = C > 4.D").
        ranks = _competition_ranks(markov_scores, final_ranking)
        groups: List[Tuple[int, List[str]]] = []
        for f_ in final_ranking:
            r = ranks[f_]
            if groups and groups[-1][0] == r:
                groups[-1][1].append(f_)
            else:
                groups.append((r, [f_]))
        f.write("\nFinal ranking (Markov): "
                + " > ".join(f"{r}.{' = '.join(fs)}" for r, fs in groups) + "\n")
        f.write("\nNote: mean|SHAP| = magnitude of the detector's influence on the meta-learner's "
                "output; mean SHAP = its net (signed) direction; both are label-free. PFI = F1 drop "
                "when the column is shuffled (label-based). A Markov-chain rank aggregation "
                "(stationary distribution over the three methods' pairwise preferences) merges the "
                "rankings; π is each detector's stationary probability (higher = stronger consensus).\n")

    result = {
        "best_ensemble": list(best_ensemble),
        "feature_names": feature_names,
        "meta_model_type": meta_model_type,
        "model_source": used_source,
        "baseline_f1": baseline_f1,
        "shap_importance": shap_abs,
        "shap_signed_importance": shap_signed,
        "pfi_importance": pfi_imp,
        "markov_scores": markov_scores,
        "final_ranking": final_ranking,
    }

    # ── Intermediate Representation (grounded LLM input; non-fatal) ─────────
    try:
        _ir = _ir_module()
        _ir.write_stage_ir(_ir.build_ga_combination_ir(dataset, entity, result),
                           dataset, entity, "ir_ga_combination")
    except Exception as e:
        logger.error(f"GA combination IR emission failed (non-fatal): {e}")

    return result
