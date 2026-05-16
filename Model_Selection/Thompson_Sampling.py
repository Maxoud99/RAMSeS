import copy
import os
import random
import traceback
from typing import List, Dict, Any
# from sklearn.metrics import f1_score, precision_recall_curve, auc
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter1d
from scipy.stats import multivariate_normal

from Metrics.Ensemble_GA import evaluate_individual_models
from Metrics.Ensemble_GA import evaluate_model_consistently
from Metrics.metrics import prauc, f1_score


def initialize_sliding_windows(data: np.ndarray, targets: np.ndarray, mask: np.ndarray, window_size: int,
                               step_size: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
    """
    Initialize data, target, and mask windows using a sliding window approach.

    Parameters:
    - data (np.ndarray): The input data array.
    - targets (np.ndarray): The target labels array.
    - mask (np.ndarray): The mask array.
    - window_size (int): The size of each window.
    - step_size (int): The step size between windows.

    Returns:
    - Tuple containing lists of data, targets, and masks windows, and the number of windows.
    """
    if data.size == 0 or targets.size == 0:
        raise ValueError("Data and targets must not be empty.")

    if window_size <= 0 or step_size <= 0:
        raise ValueError("Window size and step size must be greater than zero.")

    data_windows = []
    targets_windows = []
    masks_windows = []

    start_index = 0

    while start_index + window_size <= data.shape[1]:
        end_index = start_index + window_size
        data_windows.append(data[:, start_index:end_index])
        
        # Handle both 1D and 2D targets
        if targets.ndim == 1:
            targets_windows.append(targets[start_index:end_index])
        else:
            targets_windows.append(targets[:, start_index:end_index])
        
        masks_windows.append(mask[:, start_index:end_index])
        start_index += step_size

    num_windows = len(data_windows)

    return data_windows, targets_windows, masks_windows, num_windows


def sample_model(models: Dict[str, Any], means: Dict[str, np.ndarray], covariances: Dict[str, np.ndarray],
                 epsilon: float, context: np.ndarray) -> str:
    """
    Sample a model using Epsilon-Greedy or Linear Thompson Sampling strategy.

    Parameters:
    - models (Dict[str, Any]): Dictionary of models.
    - means (Dict[str, np.ndarray]): Dictionary of means for each model.
    - covariances (Dict[str, np.ndarray]): Dictionary of covariances for each model.
    - epsilon (float): Epsilon value for the Epsilon-Greedy strategy.
    - context (np.ndarray): The current context vector x (flattened data window). Used to
      compute the expected reward estimate theta_tilde^T * x for each model, which is the
      correct Linear Thompson Sampling selection criterion.

    Returns:
    - str: The chosen model name.
    """
    if random.random() < epsilon:
        chosen_model = random.choice(list(models.keys()))
        logger.info(f"Epsilon-Greedy: Randomly chosen model {chosen_model}")
    else:
        x = context.flatten()  # shape: (d,)
        samples = {}
        for model_name, mean in means.items():
            try:
                # Draw a full sample theta_tilde ~ N(mu, Sigma)
                theta_tilde = multivariate_normal.rvs(mean=mean.flatten(), cov=covariances[model_name])
                # Compute expected reward: theta_tilde^T * x  (the "Linear" in LinTS)
                samples[model_name] = float(np.dot(theta_tilde, x))
            except ValueError as e:
                logger.error(f"Error sampling model {model_name}: {e}")
                raise
        chosen_model = max(samples, key=samples.get)
        logger.info(f"Linear Thompson Sampling: Chosen model {chosen_model} with expected reward {samples[chosen_model]:.4f}")
    return chosen_model


def update_posteriors(means: Dict[str, np.ndarray], covariances: Dict[str, np.ndarray], model_name: str, reward: float,
                      features: np.ndarray) -> None:
    """
    Update the posterior means and covariances for the chosen model.

    Parameters:
    - means (Dict[str, np.ndarray]): Dictionary of means for each model.
    - covariances (Dict[str, np.ndarray]): Dictionary of covariances for each model.
    - model_name (str): The chosen model name.
    - reward (float): The reward obtained from the model evaluation.
    - features (np.ndarray): The feature vector.

    Returns:
    - None
    """
    if model_name not in means or model_name not in covariances:
        raise ValueError(f"Model name {model_name} not found in means or covariances.")

    features = features.reshape(-1, 1)  # Ensure features is a column vector
    n_features = features.shape[0]

    covariance = covariances[model_name]
    mean = means[model_name].reshape(-1, 1)

    logger.debug(f"Updating posteriors for model {model_name}")
    logger.debug(f"Features shape: {features.shape}")
    logger.debug(f"Covariance shape: {covariance.shape}")

    if covariance.shape[0] != n_features:
        logger.error(f"Shape mismatch: covariance shape {covariance.shape}, features shape {features.shape}")
        raise ValueError("Shape mismatch between covariance matrix and feature vector")

    # Sherman-Morrison rank-1 update — avoids all matrix inversions and is numerically
    # stable for any dimension (no SVD, no ill-conditioning on high-d datasets like SMD).
    #
    # Given:  Sigma_new^{-1} = Sigma^{-1} + x x^T
    # Sherman-Morrison gives Sigma_new directly:
    #   u     = Sigma @ x
    #   alpha = 1 + x^T u          (always > 0 because Sigma is PSD)
    #   Sigma_new = Sigma - (u u^T) / alpha
    #
    # Mean update derived from the same formula:
    #   mu_new = mu + u * (reward - x^T mu) / alpha
    #
    # This is mathematically identical to the double-inversion form but avoids:
    #   (a) np.linalg.inv crashing with "SVD did not converge" on ill-conditioned matrices
    #   (b) the reference bug where old_precision aliased precision in the previous code

    x = features.flatten()
    mu = mean.flatten()
    Sigma = covariance

    u = Sigma @ x                          # shape (d,)
    alpha = 1.0 + float(x @ u)            # scalar, always >= 1 when Sigma is PSD
    alpha = max(alpha, 1e-10)              # numerical safety guard

    Sigma_new = Sigma - np.outer(u, u) / alpha
    mu_new = mu + u * (reward - float(x @ mu)) / alpha

    covariances[model_name] = Sigma_new
    means[model_name] = mu_new
    logger.info(f"Updated posteriors for model {model_name}: mean = {mu_new}, alpha = {alpha:.4f}")


def calculate_reward(f1: float, pr_auc: float, f1_weight: float, pr_auc_weight: float) -> float:
    """
    Calculate the reward based on F1 score and PR AUC.

    Parameters:
    - f1 (float): F1 score.
    - pr_auc (float): Precision-Recall AUC.
    - f1_weight (float): Weight for F1 score.
    - pr_auc_weight (float): Weight for PR AUC.

    Returns:
    - float: The calculated reward.
    """
    return (f1_weight * f1) + (pr_auc_weight * pr_auc)


def fit_linear_thompson_sampling(dataset,
                                 models: Dict[str, Any], data: np.ndarray, targets: np.ndarray,
                                 initial_epsilon: float = 0.2,
                                 epsilon_decay: float = 0.99, f1_weight: float = 0.7, pr_auc_weight: float = 0.3,
                                 iterations: int = 100) -> Tuple[
    Dict[str, np.ndarray], Dict[str, np.ndarray], List[Dict[str, float]]]:
    """
    Fit models using Linear Thompson Sampling.

    Parameters:
    - dataset: Dataset object containing data and labels.
    - models (Dict[str, Any]): Dictionary of models.
    - data (np.ndarray): Input data array.
    - targets (np.ndarray): Target labels array.
    - initial_epsilon (float): Initial epsilon value for Epsilon-Greedy strategy.
    - epsilon_decay (float): Decay rate for epsilon.
    - f1_weight (float): Weight for F1 score in reward calculation.
    - pr_auc_weight (float): Weight for PR AUC in reward calculation.
    - iterations (int): Number of iterations for sampling.

    Returns:
    - Tuple containing dictionaries of means, covariances, and history of means.
    """
    mask = dataset.entities[0].mask
    print(f"Data shape before windowing: {data.shape}")
    print(f"Targets shape before windowing: {targets.shape}")
    print(f"Mask shape before windowing: {mask.shape}")

    n_times = dataset.entities[0].n_time
    dataset.entities[0].n_time = n_times // iterations
    dataset.total_time = n_times // iterations
    print(f"window size {int(np.size(targets.flatten()) / iterations)}")
    print(f"step size {int(np.size(targets.flatten()) / (2 * iterations))}")
    data_windows, targets_windows, New_mask, num_windows = initialize_sliding_windows(data, targets, mask, int(np.size(
        targets.flatten()) / iterations), int(np.size(targets.flatten()) / (2 * iterations)))

    # n_features should be the flattened window length (n_channels * window_size)
    # data_windows elements have shape (n_channels, window_size), so flatten to get full feature length
    n_features = data_windows[0].flatten().shape[0]
    means = {model_name: np.zeros((n_features, 1)) for model_name in models}
    covariances = {model_name: np.eye(n_features) for model_name in models}
    epsilon = initial_epsilon
    history = []
    list_of_chosen_models = []

    for iteration in range(num_windows):
        logger.info(f"Iteration {iteration + 1}")
        try:
            # Pass the current window as context so selection uses theta_tilde^T * x
            # Normalise to unit length so that datasets with large sensor values
            # (e.g. SMD with 38 channels) do not cause xxᵀ to explode and collapse Σ.
            context = data_windows[iteration].flatten()
            context_norm = np.linalg.norm(context)
            context = context / (context_norm + 1e-10)
            chosen_model_name = sample_model(models, means, covariances, epsilon, context)
        except ValueError as e:
            logger.error(f"Error sampling model: {e}")
            continue  # Skip to the next iteration on error
        chosen_model = models[chosen_model_name]
        list_of_chosen_models.append(chosen_model_name)

        X_test_window = data_windows[iteration]
        y_test_window = targets_windows[iteration]
        masks_window = New_mask[iteration]

        dataset.entities[0].Y = X_test_window
        dataset.entities[0].labels = targets_windows[iteration]
        dataset.entities[0].mask = masks_window
        dataset.entities[0].n_time = np.size(targets_windows[iteration].flatten())
        dataset.total_time = np.size(targets_windows[iteration].flatten())

        try:
            y_true, y_scores, y_true_dict, y_scores_dict = evaluate_model_consistently(dataset, chosen_model,
                                                                                       chosen_model_name)

            # _, _, f1, pr_auc, *_ = range_based_precision_recall_f1_auc(y_true, y_scores)
            f1, precision, recall, TP, TN, FP, FN = f1_score(y_scores, y_true)
            # f1, precision, recall, TP, TN, FP, FN = f1_soft_score(y_scores, y_true)
            # f1 = get_composite_fscore_raw(y_scores, y_true)

            pr_auc = prauc(y_true, y_scores)
            reward = calculate_reward(f1, pr_auc, f1_weight, pr_auc_weight)
            # Normalise features to unit length — must match the normalisation applied
            # to context in sample_model so that θ̃ᵀx (selection) and the posterior
            # update operate in the same feature space.
            features = X_test_window.flatten()
            features = features / (np.linalg.norm(features) + 1e-10)

            # Log the actual and expected feature vector sizes
            logger.debug(f"Expected feature vector size: {n_features}, actual feature vector size: {features.shape[0]}")

            if features.shape[0] != n_features:
                raise ValueError(
                    f"Feature vector size mismatch: expected {n_features}, got {features.shape[0]}. "
                    "This should not happen after the n_features fix — check data shapes.")

            logger.debug(f"Feature vector shape after adjustment: {features.shape}")
            logger.debug(f"Covariance matrix shape: {covariances[chosen_model_name].shape}")

            update_posteriors(means, covariances, chosen_model_name, reward, features)
            logger.info(
                f"Window {iteration + 1}: Model {chosen_model_name} - F1 Score = {f1}, PR AUC = {pr_auc}, Reward = {reward}")
            logger.info(f"Means: {means}")
            logger.info(f"Covariances: {covariances}")

        except Exception as e:
            logger.error(f"Error evaluating model {chosen_model_name}: {e}")
            detailed_traceback = traceback.format_exc()
            print(detailed_traceback)
            continue  # Skip the current iteration on error

        epsilon *= epsilon_decay

        history.append({model_name: means[model_name].flatten() for model_name in models})
        logger.info(f"Finished iteration {iteration + 1}")

    return means, covariances, history, list_of_chosen_models


def rank_models(means: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
    """
    Rank the models based on their mean vectors.

    Parameters:
    - means (Dict[str, np.ndarray]): Dictionary of mean vectors for each model.

    Returns:
    - List[Tuple[str, float]]: List of models and their mean scores, sorted from highest to lowest.
    """
    model_ranking = {model_name: np.dot(mean.flatten(), mean.flatten()) for model_name, mean in means.items()}
    ranked_models = sorted(model_ranking.items(), key=lambda x: x[1], reverse=True)
    return ranked_models


def calculate_score(mean: np.ndarray) -> float:
    """
    Calculate the score for a given mean vector.

    Parameters:
    - mean (np.ndarray): Mean vector.

    Returns:
    - float: Score.
    """
    return np.dot(mean.flatten(), mean.flatten())


def plot_history(history: List[Dict[str, np.ndarray]], models: Dict[str, Any],
                 dataset: str, entity: str, iterations: int) -> None:
    """
    Plot the history of model scores over time with academic styling.

    Parameters:
    - history (List[Dict[str, np.ndarray]]): List of mean vectors per iteration.
    - models (Dict[str, Any]): Dictionary of model names to model objects.
    - dataset (str): Dataset name.
    - entity (str): Entity name.
    - iterations (int): Number of iterations.

    Returns:
    - None
    """
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for model_name in models.keys():
        raw_scores = [calculate_score(h[model_name]) for h in history]
        smoothed_scores = gaussian_filter1d(raw_scores, sigma=2)  # Set sigma=0 to disable smoothing
        ax.plot(range(len(history)), smoothed_scores, label=model_name, linewidth=1.4)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Model Score Trajectories Over Iterations')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc='upper left', ncol=2, frameon=False)
    plt.tight_layout(pad=1.2)

    # Save as high-resolution PNG
    directory = f'myresults/Thomposon/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'{directory}/history_plot_{iterations}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_models_scores(algorithm_list, test_data, y_scores_list, dataset, entity, iterations, F1_Score_list_ind_curent,
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
        # axes[i + 2].vlines(spike_indices, ymin=0, ymax=1, color='red', label='Detected Anomalies')

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
    plt.savefig(f'{directory}/performance_plot_{iterations}.png')
    # plt.show()


def run_linear_thompson_sampling(test_data, trained_models, model_names, dataset, entity, iterations, iteration,
                                 initial_epsilon=0.2, epsilon_decay=0.99, f1_weight=0.5, pr_auc_weight=0.5):
    """
    Run the entire Linear Thompson Sampling process.

    Parameters:
    - test_data: The dataset to test on.
    - trained_models: Dictionary of trained models.
    - model_names: List of model names.
    - dataset (str): Name of the dataset.
    - entity (str): Name of the entity.
    - iterations (int): Number of iterations for sampling.
    - initial_epsilon (float): Initial epsilon value for Epsilon-Greedy strategy.
    - epsilon_decay (float): Decay rate for epsilon.
    - f1_weight (float): Weight for F1 score in reward calculation.
    - pr_auc_weight (float): Weight for PR AUC in reward calculation.

    Returns:
    - None
    """
    test_data_copy = copy.deepcopy(test_data)
    means, covariances, history, list_of_chosen_models = fit_linear_thompson_sampling(
        test_data,
        trained_models,
        test_data.entities[0].Y,
        test_data.entities[0].labels,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        f1_weight=f1_weight,
        pr_auc_weight=pr_auc_weight,
        iterations=iterations,
    )

    # Rank models
    ranked_models = rank_models(means)

    directory = f'myresults/Thomposon/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(directory, f"thompson_sampling_results_{dataset}_{entity}_{iterations}_{iteration}.txt")

    # Plot history
    plot_history(history, trained_models, dataset, entity, iterations)
    # evaulate over all current data and other data

    model_names = [model[0] for model in ranked_models]

    print("Thompson Sampling Results")
    print(model_names)
    print("Over the current one")
    
    # Skip redundant evaluation - models were already evaluated in GA stage
    # individual_predictions, adjusted_y_pred_ind_current, F1_Score_list_ind_curent, PR_AUC_Score_list_ind_curent = evaluate_individual_models(
    #     model_names, test_data_copy, trained_models)
    # plot_models_scores(model_names, test_data_copy, adjusted_y_pred_ind_current, dataset, entity, iterations,
    #                    F1_Score_list_ind_curent, PR_AUC_Score_list_ind_curent)
    #
    # individual_predictions, false_rate, F1_Score_list_ind_curent, PR_AUC_Score_list_ind_curent = evaluate_individual_models_regular_f1_prauc(
    #     model_names, test_data_copy, trained_models)

    # Simplified output - skip detailed misclassification analysis
    # misclassified_current = []
    # for predicts in adjusted_y_pred_ind_current:
    #     true_values = np.array(test_data_copy.entities[0].labels)
    #     predicted_values = np.array(predicts)
    #     predicted_int = predicted_values.astype(int)
    #     incorrect_predictions = predicted_int != true_values
    #     misclassified_count = np.sum(incorrect_predictions)
    #     misclassified_current.append(misclassified_count)
    #
    # f1_models_curent = {}
    # pr_models_curent = {}
    # i = 0
    # for model_name in model_names:
    #     f1_models_curent[model_name] = F1_Score_list_ind_curent[i]
    #     pr_models_curent[model_name] = PR_AUC_Score_list_ind_curent[i]
    #     i += 1

    # Write summary and rankings to a file (simplified)
    with open(output_file, 'w') as f:
        f.write("Summary of Linear Thompson Sampling:\n")
        for model_name, mean in means.items():
            f.write(f"Model: {model_name}\n")
            f.write(f"  Mean: {mean}\n")
        f.write(f"choses models for each round\n")
        f.write(f"{list_of_chosen_models}\n")

        f.write("\nModels ranked by mean score:\n")
        for rank, (model_name, score) in enumerate(ranked_models, 1):
            f.write(f"{rank}. {model_name} with score {score}\n")
        # Skip detailed evaluation metrics to save computation time
        # f.write("\n evaluation for models over the current test data:\n")
        # f.write(f"{misclassified_current}")
        # f.write("\n f1_score list for models over the current test data:\n")
        # f.write(f"{f1_models_curent}")
        # f.write("\n pr_score list for models over the current test data:\n")
        # f.write(f"{pr_models_curent}")

    print(f"Results saved to {output_file}")
    return model_names
