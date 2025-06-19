import copy
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from Metrics.metrics import range_based_precision_recall_f1_auc
from Utils.model_selection_utils import evaluate_model


def intersperse_borderline_normal_points(data, labels, factor, min_scale=0.95, max_scale=1.05):
    """
    Intersperse new borderline normal points throughout the dataset by adding scaled noise
    based on local standard deviation. Data is expected to have features as the first dimension
    and samples as the second dimension.

    Args:
        data (np.ndarray): Original dataset (2D: [n_features, n_samples]).
        labels (np.ndarray): Original labels of the data (1D: [n_samples]).
        factor (float): Fraction of the original samples count to determine number of new points.
        min_scale (float): Minimum multiplier for the local standard deviation.
        max_scale (float): Maximum multiplier for the local standard deviation.
        contextual_length (int): Number of points to consider for local statistics.

    Returns:
        tuple: Dataset including the new borderline normal points interspersed, and their corresponding labels.
    """
    n_features, n_samples = data.shape
    augmented_data = []
    augmented_labels = []
    injected_normal_indices = []
    injected_anomaly_indices = []
    # Calculate how often to insert a new point
    num_new_points = int(factor * n_samples)
    contextual_length = int(0.05 * factor * n_samples)
    insert_every = n_samples // num_new_points

    new_point_counter = 0

    for i in range(n_samples):
        # Append original data point
        augmented_data.append(data[:, i])
        augmented_labels.append(labels[i])

        # Check if it's time to insert a new borderline normal point
        if new_point_counter < num_new_points and (i % insert_every == 0 or i == n_samples - 1):
            new_data = np.zeros(n_features)
            for j in range(n_features):
                # Calculate local standard deviation within a contextual window
                start_idx = max(0, i - contextual_length)
                end_idx = min(n_samples, i + contextual_length + 1)
                local_std = np.std(data[j, start_idx:end_idx])

                # Determine scaling factor for this new point
                scale_factor = np.random.uniform(min_scale, max_scale)

                # Generate noise
                noise = np.random.normal(0, local_std * scale_factor)
                new_data[j] = noise  # Create a new point by adding noise to the base point

            # Add new point
            augmented_data.append(new_data)
            # Label the point based on the scale factor used
            new_label = 1 if scale_factor > 1.0 else 0
            augmented_labels.append(new_label)
            if new_label == 0:
                injected_normal_indices.append(len(augmented_data) - 1)
            else:
                injected_anomaly_indices.append(len(augmented_data) - 1)
            new_point_counter += 1

    # Convert lists back to numpy arrays with correct shape
    augmented_data = np.array(augmented_data).T  # Transpose to match original data shape
    augmented_labels = np.array(augmented_labels)

    return augmented_data, augmented_labels, injected_normal_indices, injected_anomaly_indices


def plot_data_with_injected_points(original_data, augmented_data, injected_normal_indices, injected_anomaly_indices,
                                   dataset, entity, feature_index=0):
    fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    # Plot the original data on the first subplot
    axes[0].plot(original_data[feature_index, :], color='darkblue', linestyle='--', label='Original Data')
    axes[0].set_title(f'Original Data for Feature {feature_index}')
    axes[0].set_ylabel('Feature Value')
    axes[0].legend()
    axes[0].grid(True)

    # Plot the augmented data on the second subplot
    axes[1].plot(augmented_data[feature_index, :], color='lightblue', label='Augmented Data')

    # Highlight injected normal points in green
    if injected_normal_indices:
        axes[1].scatter(injected_normal_indices,
                        [augmented_data[feature_index, idx] for idx in injected_normal_indices],
                        color='green', label='Injected Normal Points', marker='o', s=50)

    # Highlight injected anomaly points in red
    if injected_anomaly_indices:
        axes[1].scatter(injected_anomaly_indices,
                        [augmented_data[feature_index, idx] for idx in injected_anomaly_indices],
                        color='red', label='Injected Anomaly Points', marker='x', s=50)

    axes[1].set_title(f'Augmented Data with Injected Points for Feature {feature_index}')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Feature Value')
    axes[1].legend()
    axes[1].grid(True)
    # Specify the directory
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory = f'myresults/robustness/off_by/{dataset}/{entity}/'
    filename = f'Data_vs_DataWithAnomalies_{date_time_string}_.png'
    full_path = os.path.join(directory, filename)

    # Check if the directory exists, and if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Save the figure
    plt.savefig(full_path, dpi=300)  # Save as PNG file with high resolution

    plt.show()


def run_off_by_threshold(test_data, trained_models, model_names, dataset, entity):
    dataSet_before = copy.deepcopy(test_data)
    data = test_data.entities[0].Y
    labels = test_data.entities[0].labels
    factor = .1
    augmented_data, augmented_labels, injected_normal_indices, injected_anomaly_indices = intersperse_borderline_normal_points(
        data, labels, factor)
    test_data.entities[0].Y = augmented_data
    test_data.entities[0].labels = augmented_labels
    n_times = test_data.entities[0].n_time
    test_data.total_time = int(n_times * (factor + 1))
    test_data.entities[0].n_time = int(n_times * (factor + 1))
    test_data.entities[0].mask = np.ones(test_data.entities[0].Y.shape)
    # original_data, augmented_data, injected_normal_indices, injected_anomaly_indices
    plot_data_with_injected_points(dataSet_before.entities[0].Y, test_data.entities[0].Y, injected_normal_indices,
                                   injected_anomaly_indices, dataset, entity)
    results = {}
    adjusted_y_pred_dict = {}
    for model_name in model_names:
        model = trained_models.get(model_name)
        results[model_name] = []
        adjusted_y_pred_dict[model_name] = []
        if model:
            evaluation = evaluate_model(test_data, model, model_name)  # Assume this function returns a dict
            y_true = evaluation['anomaly_labels'].flatten()
            y_scores = evaluation['entity_scores'].flatten()
            _, _, best_f1, pr_auc, adjusted_y_pred = range_based_precision_recall_f1_auc(y_true, y_scores)
            adjusted_y_pred_dict[model_name].append(adjusted_y_pred)
            results[model_name].append({'f1': best_f1, 'pr_auc': pr_auc})
            logger.info(f"Evaluated {model_name}: F1={best_f1}, PR_AUC={pr_auc}")

    ranked_by_f1 = sorted(results.items(), key=lambda x: x[1][0]['f1'], reverse=True)
    ranked_by_f1_names = [item[0] for item in ranked_by_f1]
    ranked_by_pr_auc = sorted(results.items(), key=lambda x: x[1][0]['pr_auc'], reverse=True)
    ranked_by_pr_auc_names = [item[0] for item in ranked_by_pr_auc]

    true_values = np.array(test_data.entities[0].labels)  # 1 for anomaly, 0 for normal
    print(10 * '=')
    predicted_values = np.array(adjusted_y_pred_dict[ranked_by_f1_names[0]]).flatten()  # Flatten the list of arrays

    # Converting boolean predictions to integer for easy plotting (True to 1, False to 0)
    predicted_int = predicted_values.astype(int)

    # Identifying incorrect predictions
    incorrect_predictions = predicted_int != true_values
    misclassified_count = np.sum(incorrect_predictions)  # Number of misclassifications
    total_anomalies = np.sum(true_values)  # Total number of real anomalies
    total_data = len(true_values)  # Total number of data points
    print(incorrect_predictions)
    print(misclassified_count)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, '.', label='True Values (Anomalies)', color='blue')  # Plot true values
    plt.plot(predicted_int, 'x', label='Predicted Values (Anomalies)', color='red')  # Plot predicted values
    plt.scatter(np.where(incorrect_predictions)[0], predicted_int[incorrect_predictions], facecolors='none',
                edgecolors='purple', s=100, label='Incorrect Predictions', linewidth=2)
    plt.title(
        f'True vs. Predicted Anomalies \n Misclassified Anomalies: {misclassified_count}\n Total Anomalies: {total_anomalies} \n Total Data: {total_data}')
    plt.xlabel('Index')
    plt.ylabel('Anomaly Presence')
    plt.yticks([0, 1], ['No Anomaly', 'Anomaly'])  # Set y-ticks to be explicit about what 0 and 1 represent
    plt.legend()
    plt.grid(True)

    # Specify the directory
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory = f'myresults/robustness/off_by/{dataset}/{entity}/'
    filename = f'{dataset}_{entity}_Misclassified Anomalies_{date_time_string}.png'
    full_path = os.path.join(directory, filename)

    # Check if the directory exists, and if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the figure
    plt.savefig(full_path, dpi=300)  # Save as PNG file with high resolution

    plt.show()

    return ranked_by_f1, ranked_by_pr_auc, ranked_by_f1_names, ranked_by_pr_auc_names
