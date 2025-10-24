import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from Metrics.metrics import prauc, f1_score
from Utils.model_selection_utils import evaluate_model


# class Mevaluation(object):
#     def __init__(self):
#         args = get_args_from_cmdline()
#         self.data_dir = args['dataset_path']
#         self.result_dir = args['results_path']
#         self.trained_model_path = args['trained_model_path']
#         self.overwrite = args['overwrite']
#         self.verbose = args['verbose']
#
#     def evaluate_model(self, _dataset_type: Optional[str] = None, _dataset_entity: Optional[str] = None):
#         inject_abn_types = select_inject_abn_types_by_data_entity(_data_name=_dataset_entity)
#         inject_abn_list = inject_abn_types.split('_')
#         algorithms = select_algorithms_by_data_entity(_data_name=_dataset_entity)
#         model_name_list = algorithms
#         logger.info(f'evaluate_model method inject_abn_list is {inject_abn_list}, model_name_list is {model_name_list}')
#         rank_model_params = {
#             'dataset': _dataset_type,
#             'entity': _dataset_entity,
#             'inject_abn_list': inject_abn_list,
#             'model_name_list': model_name_list,
#             'trained_model_path': self.trained_model_path,
#             'downsampling': 10,
#             'min_length': 256,
#             'root_dir': self.data_dir,
#             'normalize': True,
#             'verbose': False
#         }
#         rankingObj = RankModels(**rank_model_params)
#         return rankingObj.evaluate_models(n_repeats=1, n_neighbors=[4], split='test', synthetic_ranking_criterion='f1',
#                                           n_splits=100)


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
        y_true (np.ndarray): Array of true labels.

    Returns:
        np.ndarray: Array of true labels with injected anomalies.
    """
    num_anomalies = int(len(y_true) / 10)
    # num_anomalies = 1
    indices = np.random.choice(len(y_true), num_anomalies, replace=False)
    y_true[indices] = 1
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
    y_true_agg_dict = {}
    base_model_predictions_dict = {}
    if is_ensemble:
        y_true_agg = None
        base_model_predictions = []

        for sub_model_name in model_name:
            sub_model = model.get(sub_model_name)
            if sub_model:
                evaluation = evaluate_model(data, sub_model, sub_model_name)
                y_true = evaluation['anomaly_labels'].flatten()
                y_scores = evaluation['entity_scores'].flatten()
                base_model_predictions.append(y_scores)
                base_model_predictions_dict[sub_model_name] = y_scores
                if y_true_agg is None:
                    y_true_agg = y_true
                    y_true_agg_dict[sub_model_name] = y_true
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
            # _, _, best_f1, pr_auc, adjusted_y_pred = range_based_precision_recall_f1_auc(y_true, y_scores)
            best_f1, precision, recall, TP, TN, FP, FN = f1_score(y_scores, y_true)
            # best_f1, precision, recall, TP, TN, FP, FN = f1_soft_score(y_scores, y_true)
            # best_f1 = get_composite_fscore_raw(y_scores, y_true)
            pr_auc = prauc(y_true, y_scores)
            logger.info(f"Model {model_name}: F1 score = {best_f1}, PR AUC = {pr_auc}")
            predictions[model_name] = (y_true, y_scores)
            adjusted_y_pred_list.append(y_scores)
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
        tuple: Best F1 score, PR AUC, and fitness score.
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
        # y_true_test = inject_synthetic_anomalies(y_true_test)

    # Generate prediction scores using the meta-model
    y_scores = meta_model.predict_proba(base_model_predictions_test)[:, 1]

    # Calculate evaluation metrics: F1 score and PR AUC
    # _, _, best_f1, pr_auc, adjusted_y_pred = range_based_precision_recall_f1_auc(y_true_test, y_scores)
    best_f1, precision, recall, TP, TN, FP, FN = f1_score(y_scores, y_true_test)
    # best_f1, precision, recall, TP, TN, FP, FN = f1_soft_score(y_scores, y_true_test)
    # best_f1 = get_composite_fscore_raw(y_scores, y_true_test)
    pr_auc = prauc(y_true_test, y_scores)

    # Calculate the fitness score as the average of the F1 score and PR AUC
    # fitness = (best_f1 + pr_auc) / 2
    fitness = best_f1
    logger.info(
        f"Evaluated fitness for ensemble {ensemble} with F1 score {best_f1} and PR AUC {pr_auc}, resulting in fitness {fitness}")
    return best_f1, pr_auc, fitness, y_scores, y_true_test


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
                      population_size, generations, mutation_rate):
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
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_name = f'ensemble_scores_{dataset}_{entity}_{meta_model_type}_{population_size}_{generations}_{mutation_rate}_UMS_{date_time_string}.png'
    plot_path = f'myresults/Outputs/GA_Ens/{dataset}/{entity}'
    plot_scores_vs_true(test_data, F1_Score_list_ind, PR_AUC_Score_list_ind, adjusted_y_pred_ind, algorithm_list,
                        plot_name, plot_path)
    # individual_predictions = []
    population = initialize_population(algorithm_list, population_size)
    print(population)
    evaluated_ensembles = {}  # HashMap to track evaluated ensembles and their scores
    file_name = f'myresults/Outputs/GA_Ens/ensemble_scores_{dataset}_{entity}_{meta_model_type}_{population_size}_{generations}_{mutation_rate}_{date_time_string}.txt'

    best_f1 = 0
    best_pr_auc = 0
    best_fitness = 0
    best_ensemble = None
    adjusted_y_pred_list = []
    F1_Score_list = []
    list_ensemble = []
    PR_AUC_Score_list = []
    fitness_list = []
    y_true_train, base_model_predictions_train, y_true_train_dict, base_model_predictions_train_dict = evaluate_model_consistently(
        train_data,
        trained_models,
        algorithm_list,
        is_ensemble=True)
    y_true_test, base_model_predictions_test, y_true_test_dict, base_model_predictions_test_dict = evaluate_model_consistently(
        test_data, trained_models, algorithm_list,
        is_ensemble=True)
    print("y_true_train mine")
    print(y_true_train)
    for generation in range(generations):
        logger.info(f"Generation {generation + 1}")
        print(f"Generation {generation + 1}")

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

        population = new_population

        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > best_fitness:
            best_f1 = f1_scores[best_idx]
            best_pr_auc = pr_aucs[best_idx]
            best_fitness = fitness_scores[best_idx]
            best_ensemble = population[best_idx]

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

    return best_ensemble, best_f1, best_pr_auc, best_fitness, individual_predictions, base_model_predictions_train, base_model_predictions_test, y_true_train, y_true_test, meta_model_type

# Usage
# Assuming train_data and test_data are already loaded and preprocessed
# algorithm_list = ['LOF', 'NN', 'RNN']
# trained_models = {'LOF': lof_model, 'NN': nn_model, 'RNN': rnn_model}
# best_ensemble, best_f1, best_pr_auc, best_fitness = genetic_algorithm(train_data, test_data, algorithm_list, trained_models, meta_model_type='lr')
# You can change meta_model_type to 'rf', 'gbm', or 'svm' for Random Forest, Gradient Boosting Machine, and SVM respectively
