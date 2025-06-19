# time_series_framework/app.py
import os

from Datasets.load import load_data
from Utils.utils import get_args_from_cmdline
from Model_Optimization.Optimize import run_optimization
from loguru import logger
import traceback

# def run_app():
#     args = get_args_from_cmdline()
#     data_dir = args['dataset_path']
#     train_data = load_data(dataset='smd', group='train', entities='machine-1-1', downsampling=10,
#                            min_length=256, root_dir=data_dir, normalize=True, verbose=False)
#     test_data = load_data(dataset='smd', group='test', entities='machine-1-1', downsampling=10,
#                           min_length=256, root_dir=data_dir, normalize=True, verbose=False)
#
#     try:
#         # Run optimization
#         best_params = run_optimization(train_data, test_data)
#         print(f"Best hyperparameters: {best_params}")
#
#         # Further actions can be taken with best_params if needed
#     except:
#         logger.info(f'Traceback for Entity: machine 1-1 Dataset: SMD')
#         print(traceback.format_exc())
#
# if __name__ == "__main__":
#     run_app()

# ------------------------------------
# time_series_framework/app.py
save_dir = "D:/Master/SS_2024_Thesis_ISA/Thesis/Work-docs/RAMS-TSAD/Mononito/trained_models/smd/machine-1-1/"
# time_series_framework/app.py


# time_series_framework/app.py
import os
import logging
import torch as t
from Datasets.load import load_data
from Utils.utils import get_args_from_cmdline
from Model_Training.train import TrainModels
from loguru import logger
import traceback
from Utils.model_selection_utils import evaluate_model
# Import the ensemble and genetic algorithm functions
from Metrics.Ens_GA import genetic_algorithm

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
                model = t.load(f)
                model.eval()  # Set model to evaluation mode
                trained_models[model_name] = model
        else:
            raise FileNotFoundError(f"Model {model_name} not found in {save_dir}")
    return trained_models

def evaluate_ensemble(ensemble, data, models_dict, eval_batch_size=128):
    """Evaluate the ensemble of models."""
    predictions = []
    for model_name, weight in ensemble.items():
        model = models_dict[model_name]
        model_predictions = evaluate_model(data=data, model=model, model_name=model_name, padding_type='right', eval_batch_size=eval_batch_size)
        predictions.append(model_predictions['entity_scores'] * weight)

    # Aggregate predictions
    ensemble_scores = sum(predictions) / len(predictions)

    return ensemble_scores

def run_app():
    args = get_args_from_cmdline()
    data_dir = args['dataset_path']
    train_data = load_data(dataset='smd', group='train', entities='machine-1-1', downsampling=10,
                           min_length=256, root_dir=data_dir, normalize=True, verbose=False)
    test_data = load_data(dataset='smd', group='test', entities='machine-1-1', downsampling=10,
                          min_length=256, root_dir=data_dir, normalize=True, verbose=False)

    # Ensure data is correctly loaded
    if not train_data.entities:
        logger.error("Failed to load training data. Please check the dataset and paths.")
        return

    if not test_data.entities:
        logger.error("Failed to load test data. Please check the dataset and paths.")
        return

    model_trainer = TrainModels(dataset='smd',
                                entity='machine-1-1',
                                algorithm_list=['LOF', 'NN', 'RNN'],
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
        # save_dir = args['trained_model_path']
        algorithm_list = ['LOF_1', 'LOF_2', 'LOF_3', 'LOF_4', 'NN_1', 'NN_2', 'NN_3', 'RNN_1', 'RNN_2', 'RNN_3', 'RNN_4']
        models_dict = load_trained_models(algorithm_list, save_dir)

        if not models_dict:
            raise ValueError("No models were loaded. Please check the model paths and ensure models are trained.")

        # Run the genetic algorithm to find the best ensembles using the specified technique
        best_ensembles = genetic_algorithm(models_dict, train_data, test_data, num_generations=10, population_size=10, metric='pr_auc')

        # Log or save the results
        for ensemble, score in best_ensembles:
            logger.info(f'Ensemble: {ensemble}, Score: {score}')

    except Exception as e:
        logger.info(f'Traceback for Entity: machine 1-1 Dataset: SMD')
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_app()

# =================================================================================================================================================
import random
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from loguru import logger
from typing import Optional,List
from Datasets.load import load_data
from Model_Selection.inject_anomalies import InjectAnomalies
from Utils.model_selection_utils import evaluate_model
from Utils.utils import get_args_from_cmdline
from Utils.utils import visualize_data
from dao.mdata.mdata import update_data_status_by_name, select_algorithms_by_data_entity, \
    select_inject_abn_types_by_data_entity
from Model_Selection.model_selection import RankModels
from Metrics.metrics import mse, mae, smape, mape, prauc, gaussian_likelihood, get_range_vus_roc, best_f1_linspace, \
    adjusted_precision_recall_f1_auc, sequence_precision_delay, range_based_precision_recall_f1_auc, \
    calculate_mutual_information, calculate_cdi

class Mevaluation(object):
    def __init__(self):
        args = get_args_from_cmdline()
        self.data_dir = args['dataset_path']
        self.result_dir = args['results_path']
        self.trained_model_path = args['trained_model_path']
        self.overwrite = args['overwrite']
        self.verbose = args['verbose']

    def evaluate_model(self, _dataset_type: Optional[str] = None, _dataset_entity: Optional[str] = None):
        inject_abn_types = select_inject_abn_types_by_data_entity(_data_name=_dataset_entity)
        inject_abn_list = inject_abn_types.split('_')
        algorithms = select_algorithms_by_data_entity(_data_name=_dataset_entity)
        model_name_list = algorithms
        logger.info(f'evaluate_model method inject_abn_list is {inject_abn_list}, model_name_list is {model_name_list}')
        rank_model_params = {
            'dataset': _dataset_type,
            'entity': _dataset_entity,
            'inject_abn_list': inject_abn_list,
            'model_name_list': model_name_list,
            'trained_model_path': self.trained_model_path,
            'downsampling': 10,
            'min_length': 256,
            'root_dir': self.data_dir,
            'normalize': True,
            'verbose': False
        }
        rankingObj = RankModels(**rank_model_params)
        return rankingObj.evaluate_models(n_repeats=1, n_neighbors=[4], split='test', synthetic_ranking_criterion='f1',
                                          n_splits=100)


def initialize_population(algorithm_list, population_size):
    population = []
    for _ in range(population_size):
        ensemble_size = random.randint(1, len(algorithm_list))
        ensemble = random.sample(algorithm_list, k=ensemble_size)
        population.append(ensemble)
    logger.info(f"Initialized population with {population_size} ensembles")
    return population


def calculate_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred)


def calculate_pr_auc(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


def inject_synthetic_anomalies(y_true, num_anomalies=10):
    indices = np.random.choice(len(y_true), num_anomalies, replace=False)
    y_true[indices] = 1
    return y_true


def train_meta_model(base_model_predictions, y_true):
    meta_model = LogisticRegression()
    meta_model.fit(base_model_predictions, y_true)
    return meta_model


def f1_soft_score(predict, actual):
    # Predict: 1/0
    # Actual: [0,1]
    actual = actual / np.max(actual)

    negatives = 1 * (actual == 0)

    TP = np.sum(predict * actual)  # weighted by actual
    TN = np.sum((1 - predict) * (negatives))
    FP = np.sum(predict * (negatives))
    FN = np.sum((1 - predict) * actual)  # weighted by actual
    precision = TP / (TP + FP * np.mean(actual[actual > 0]) + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)

    return f1, precision, recall, TP, TN, FP, FN


def best_f1_linspace(scores, labels, n_splits=100, segment_adjust=False, f1_type='standard'):
    best_threshold = 0
    best_f1 = 0
    thresholds = np.linspace(scores.min(), scores.max(), n_splits)

    if np.sum(labels) > 0:
        for threshold in thresholds:
            predict = scores >= threshold
            if f1_type == 'standard':
                f1 = f1_score(labels, predict)
            elif f1_type == 'soft':
                f1 = f1_soft_score(labels, predict)  # Assuming f1_soft_score is defined

            if f1 > best_f1:
                best_threshold = threshold
                best_f1 = f1
    else:
        best_threshold = scores.max() + 1
        best_f1 = 1

    predict = scores >= best_threshold
    if f1_type == 'standard':
        f1 = f1_score(labels, predict)
    elif f1_type == 'soft':
        f1 = f1_soft_score(labels, predict)

    return best_threshold, best_f1


def evaluate_individual_models(algorithm_list, test_data, trained_models):
    """Evaluate individual models and print their F1 score and PR AUC using dynamic thresholding."""
    for model_name in algorithm_list:
        model = trained_models.get(model_name)
        if model:
            evaluation = evaluate_model(test_data, model, model_name)
            y_true = evaluation['anomaly_labels'].flatten()
            y_scores = evaluation['entity_scores'].flatten()
            _, _, best_f1, pr_auc, *_ = range_based_precision_recall_f1_auc(y_true,y_scores)
            # y_pred = (y_scores >= best_threshold).astype(int)
            # pr_auc = calculate_pr_auc(y_true, y_scores)
            logger.info(
                f"Model {model_name}: F1 score = {best_f1}, PR AUC = {pr_auc}")


def fitness_function(ensemble, train_data, test_data, trained_models):
    base_model_predictions_train = []
    y_true_train = None

    for model_name in ensemble:
        model = trained_models.get(model_name)
        if model:
            evaluation_train = evaluate_model(train_data, model, model_name)
            base_model_predictions_train.append(evaluation_train['entity_scores'].flatten())
            if y_true_train is None:
                y_true_train = evaluation_train['anomaly_labels'].flatten()

    if not base_model_predictions_train:
        return 0, 0, 0

    base_model_predictions_train = np.array(base_model_predictions_train).T

    if len(np.unique(y_true_train)) < 2:
        logger.warning(f"Ensemble {ensemble} has only one class in the training labels. Injecting synthetic anomalies.")
        y_true_train = inject_synthetic_anomalies(y_true_train)

    meta_model = train_meta_model(base_model_predictions_train, y_true_train)

    base_model_predictions_test = []
    y_true_test = None

    for model_name in ensemble:
        model = trained_models.get(model_name)
        if model:
            evaluation_test = evaluate_model(test_data, model, model_name)
            base_model_predictions_test.append(evaluation_test['entity_scores'].flatten())
            if y_true_test is None:
                y_true_test = evaluation_test['anomaly_labels'].flatten()

    if not base_model_predictions_test:
        return 0, 0, 0

    base_model_predictions_test = np.array(base_model_predictions_test).T

    if len(np.unique(y_true_test)) < 2:
        logger.warning(f"Ensemble {ensemble} has only one class in the test labels. Injecting synthetic anomalies.")
        y_true_test = inject_synthetic_anomalies(y_true_test)

    y_scores = meta_model.predict_proba(base_model_predictions_test)[:, 1]

    print("Distribution of Prediction Scores (y_scores):", np.histogram(y_scores, bins=10))

    _, _, best_f1, pr_auc, *_ = range_based_precision_recall_f1_auc(y_true_test, y_scores)

    # y_pred = (y_scores >= best_threshold).astype(int)

    print("True Labels (y_true_test):", y_true_test)
    # print("Predictions (y_pred):", y_pred)
    print("Prediction Scores (y_scores):", y_scores)

    if np.sum(y_true_test) == 0:
        print("Warning: No positive samples in true labels (y_true_test).")

    f1 = best_f1


    fitness = (f1 + pr_auc) / 2

    logger.info(
        f"Evaluated fitness for ensemble {ensemble} with F1 score {f1} and PR AUC {pr_auc}, resulting in fitness {fitness}")
    return f1, pr_auc, fitness


def selection(population, fitness_scores, num_selected):
    selected_indices = np.argsort(fitness_scores)[-num_selected:]
    selected = [population[i] for i in selected_indices]
    logger.info(f"Selected top {num_selected} ensembles with scores {fitness_scores}")
    return selected


def crossover(parent1, parent2):
    crossover_point1 = random.randint(1, len(parent1))
    crossover_point2 = random.randint(1, len(parent2))
    child = parent1[:crossover_point1] + parent2[crossover_point2:]
    child = list(set(child))
    logger.info(f"Crossover parents {parent1} and {parent2} to create child {child}")
    return child


def mutate(ensemble, mutation_rate, algorithm_list):
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
        if len(mutated_ensemble) > 1 and random.random() > 0.5:
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

    return mutated_ensemble


def genetic_algorithm(train_data, test_data, algorithm_list, trained_models, population_size=4, generations=4,
                      mutation_rate=0.1):
    mevaluation_instance = Mevaluation()
    mevaluation_instance.evaluate_model(train_data, test_data)  # Evaluate individual models before GA

    evaluate_individual_models(algorithm_list, test_data, trained_models)

    population = initialize_population(algorithm_list, population_size)
    best_f1 = 0
    best_pr_auc = 0
    best_fitness = 0
    best_ensemble = None

    for generation in range(generations):
        logger.info(f"Generation {generation + 1}")
        print(f"Generation {generation + 1}")

        fitness_results = [fitness_function(ensemble, train_data, test_data, trained_models) for ensemble in population]
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

    logger.info(
        f"Best ensemble found: {best_ensemble} with F1 score {best_f1}, PR AUC {best_pr_auc}, and fitness {best_fitness}")
    print(
        f"Best ensemble found: {best_ensemble} with F1 score {best_f1}, PR AUC {best_pr_auc}, and fitness {best_fitness}")

    return best_ensemble, best_f1, best_pr_auc, best_fitness

# Usage
# Assuming train_data and test_data are already loaded and preprocessed
# algorithm_list = ['LOF', 'NN', 'RNN']
# trained_models = {'LOF': lof_model, 'NN': nn_model, 'RNN': rnn_model}
# best_ensemble, best_f1, best_pr_auc, best_fitness = genetic_algorithm(train_data, test_data, algorithm_list, trained_models)
# print(f'test_data.entities[0].Y: \n {np.size(test_data.entities[0].Y)}')
# print(f'test_data.entities[0].labels: \n {test_data.entities[0].labels}')
# print(f'test_data.entities[0].X: \n {test_data.entities[0].X}')
# print(f'test_data.entities[0].n_time : \n {test_data.entities[0].n_time}')
# print(f'test_data.entities[0].mask : \n {np.size(test_data.entities[0].mask)}')
# print(f'test_data.entities[0].verbose : \n {test_data.entities[0].verbose}')
# print(f'test_data.entities[0].n_exogenous : \n {test_data.entities[0].n_exogenous}')
# print(f'test_data.entities[0].n_features : \n {test_data.entities[0].n_features}')

# ----------------------------------------------------------------------------
# ///////////////////////////////////////////////////////////////////////////////



import random
import numpy as np
from scipy.stats import beta
from loguru import logger
from Algorithms.base_model import PyMADModel
from sklearn.metrics import f1_score, precision_recall_curve, auc
from typing import Tuple, Union, List
from Loaders.loader import Loader
from Datasets.dataset import Dataset, Entity
from Metrics.Ensemble_GA import evaluate_model_consistently
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from Utils.model_selection_utils import evaluate_model
from Metrics.metrics import range_based_precision_recall_f1_auc
import traceback
from Utils.utils import de_unfold
import torch as t

from typing import List, Tuple
import numpy as np


def initialize_windows(data: np.ndarray, targets: np.ndarray, mask: np.ndarray, n_windows: int) -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    if data.size == 0 or targets.size == 0:
        raise ValueError("Data and targets must not be empty.")

    if n_windows <= 0:
        raise ValueError("Number of windows must be greater than zero.")

    total_size = data.shape[1]  # We use shape[1] since the data has shape (1, n)
    window_size = total_size // n_windows
    remainder = total_size % n_windows

    data_windows = []
    targets_windows = []
    masks_windows = []

    start_index = 0

    for i in range(n_windows):
        end_index = start_index + window_size + (1 if i < remainder else 0)
        data_windows.append(data[:, start_index:end_index])
        targets_windows.append(targets[:, start_index:end_index])
        masks_windows.append(mask[:, start_index:end_index])
        start_index = end_index

    return data_windows, targets_windows, masks_windows


def sample_model(models: Dict[str, Any], priors: Dict[str, List[float]], epsilon: float) -> str:
    if random.random() < epsilon:
        chosen_model = random.choice(list(models.keys()))
        logger.info(f"Epsilon-Greedy: Randomly chosen model {chosen_model}")
    else:
        samples = {model_name: beta.rvs(a=prior[0], b=prior[1])
                   for model_name, prior in priors.items()}
        chosen_model = max(samples, key=samples.get)
        logger.info(f"Thompson Sampling: Chosen model {chosen_model} with sample value {samples[chosen_model]}")
    return chosen_model


def update_posteriors(priors: Dict[str, List[float]], model_name: str, reward: float) -> None:
    if model_name not in priors:
        raise ValueError(f"Model name {model_name} not found in priors.")

    priors[model_name][0] += reward
    priors[model_name][1] += (1 - reward)
    logger.info(f"Updated priors for model {model_name}: {priors[model_name]}")


def calculate_reward(f1: float, pr_auc: float, f1_weight: float, pr_auc_weight: float) -> float:
    return (f1_weight * f1) + (pr_auc_weight * pr_auc)


def fit_thompson_sampling(dataset,
                          models: Dict[str, Any], data: np.ndarray, targets: np.ndarray, initial_epsilon: float = 0.1,
                          epsilon_decay: float = 0.99, f1_weight: float = 0.5, pr_auc_weight: float = 0.5,
                          iterations: int = 100) -> Tuple[Dict[str, List[float]], List[Dict[str, float]]]:
    mask = dataset.entities[0].mask
    print(f"Data shape before windowing: {data.shape}")
    print(f"Targets shape before windowing: {targets.shape}")
    print(f"Mask shape before windowing: {mask.shape}")

    n_times = dataset.entities[0].n_time
    dataset.entities[0].n_time = n_times // iterations
    dataset.total_time = n_times // iterations
    data_windows, targets_windows, New_mask = initialize_windows(data, targets, mask, iterations)

    priors = {model_name: [1, 1] for model_name in models}
    epsilon = initial_epsilon
    history = []

    for iteration in range(iterations):
        logger.info(f"Iteration {iteration + 1}")
        chosen_model_name = sample_model(models, priors, epsilon)
        chosen_model = models[chosen_model_name]

        X_test_window = data_windows[iteration]
        y_test_window = targets_windows[iteration]
        masks_window = New_mask[iteration]

        print(f"First window data shape: {X_test_window[0].shape}")
        print(f"First window target shape: {y_test_window[0].shape}")
        print(f"First window mask shape: {masks_window[0].shape}")

        dataset.entities[0].Y = X_test_window
        dataset.entities[0].labels = targets_windows[iteration]
        dataset.entities[0].mask = masks_window

        print(f'test_data.entities[0].Y: \n {dataset.entities[0].Y}')
        print(f'test_data.entities[0].labels: \n {dataset.entities[0].labels}')
        print(f'test_data.entities[0].X: \n {dataset.entities[0].X}')
        print(f'test_data.entities[0].n_time : \n {dataset.entities[0].n_time}')
        print(f'test_data.entities[0].mask : \n {dataset.entities[0].mask}')
        print(f'test_data.entities[0].verbose : \n {dataset.entities[0].verbose}')
        print(f'test_data.entities[0].n_exogenous : \n {dataset.entities[0].n_exogenous}')
        print(f'test_data.entities[0].n_features : \n {dataset.entities[0].n_features}')

        try:
            y_true, y_scores, y_true_dict, y_scores_dict = evaluate_model_consistently(dataset, chosen_model, chosen_model_name)

            _, _, f1, pr_auc, *_ = range_based_precision_recall_f1_auc(y_true, y_scores)
            reward = calculate_reward(f1, pr_auc, f1_weight, pr_auc_weight)
            update_posteriors(priors, chosen_model_name, reward)
            logger.info(
                f"Window {iteration + 1}: Model {chosen_model_name} - F1 Score = {f1}, PR AUC = {pr_auc}, Reward = {reward}")
            logger.info(f"Priors: {priors}")

        except Exception as e:
            logger.error(f"Error evaluating model {chosen_model_name}: {e}")
            detailed_traceback = traceback.format_exc()
            print(detailed_traceback)
            continue  # Skip the current iteration on error

        epsilon *= epsilon_decay

        history.append({model_name: prior[0] / (prior[0] + prior[1]) for model_name, prior in priors.items()})
        logger.info(f"Finished iteration {iteration + 1}")

    return priors, history


def rank_models(priors: Dict[str, List[float]]) -> List[Tuple[str, float]]:
    """
    Rank the models based on their Beta distribution means.

    Parameters:
    priors (Dict[str, List[float]]): Dictionary of Beta distribution priors for each model.

    Mean Calculation: For each model, the mean of the Beta distribution is calculated as α/(α+β).

    Returns:
    List[Tuple[str, float]]: List of models and their beta scores, sorted from highest to lowest.
    """
    model_ranking = {model_name: prior[0] / (prior[0] + prior[1]) for model_name, prior in priors.items()}
    ranked_models = sorted(model_ranking.items(), key=lambda x: x[1], reverse=True)
    return ranked_models


def plot_history(history: List[Dict[str, float]], models: Dict[str, Any]) -> None:
    plt.figure(figsize=(12, 8))
    for model_name in models.keys():
        scores = [h[model_name] for h in history]
        plt.plot(scores, label=model_name)
    plt.xlabel('Iteration')
    plt.ylabel('Beta Mean')
    plt.title('Evolution of Model Rankings')
    plt.legend()
    plt.show()



