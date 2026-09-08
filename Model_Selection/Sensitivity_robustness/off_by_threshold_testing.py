import copy
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from Metrics.metrics import (range_based_precision_recall_f1_auc, rank_key,
                             vus_score, vus_window)
from Model_Selection.Sensitivity_robustness.plot_retention import (
    prune_superseded, prune_timestamped)
from Utils.model_selection_utils import evaluate_model, ScoringTimeout
from Utils.pipeline_spec import DEFAULT_DECISION_METRICS, metrics_required
from Explainability import ir
from Model_Selection.Sensitivity_robustness import exclusive_win_surrogates as ews


def intersperse_borderline_normal_points(data, labels, factor, min_scale=0.95, max_scale=1.05,
                                         return_records=False):
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
        return_records (bool): Explainability opt-in. When True, additionally return a list of
            per-injected-point records ``{index, scale, local_std, label}`` as a 5th value. This
            only records values already computed in the insert branch (no extra RNG draws), so the
            first four return values — and therefore the production ranking — are byte-for-byte
            identical to the default ``return_records=False`` path.

    Returns:
        tuple: Dataset including the new borderline normal points interspersed, and their corresponding
        labels (plus the per-point records when ``return_records=True``).
    """
    point_records = []
    n_features, n_samples = data.shape

    # Safety check: ensure we have enough samples
    if n_samples < 2:
        logger.warning(f"intersperse_borderline_normal_points skipped: n_samples={n_samples} < 2")
        if return_records:
            return data, labels, [], [], point_records
        return data, labels, [], []

    augmented_data = []
    augmented_labels = []
    injected_normal_indices = []
    injected_anomaly_indices = []
    # Calculate how often to insert a new point
    num_new_points = int(factor * n_samples)

    # If no new points to add, return original data
    if num_new_points == 0:
        if return_records:
            return data, labels, injected_normal_indices, injected_anomaly_indices, point_records
        return data, labels, injected_normal_indices, injected_anomaly_indices

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
            local_stds = []  # explainability: per-feature local std at this site (mean → volatility)
            for j in range(n_features):
                # Calculate local standard deviation within a contextual window
                start_idx = max(0, i - contextual_length)
                end_idx = min(n_samples, i + contextual_length + 1)
                local_std = np.std(data[j, start_idx:end_idx])
                local_stds.append(local_std)

                # Determine scaling factor for this new point
                scale_factor = np.random.uniform(min_scale, max_scale)

                # Generate noise
                noise = np.random.normal(0, local_std * scale_factor)
                # ADDED to the point at the injection site, not assigned. Assigned,
                # every injected point was a zero-centred vector while the series
                # is MinMax-scaled to [0, 1] — so none of them landed anywhere near
                # the data, every detector called them the same way, and the stage
                # produced no exclusive wins to explain.
                new_data[j] = data[j, i] + noise

            # Add new point
            augmented_data.append(new_data)
            # Label the point based on the scale factor used
            new_label = 1 if scale_factor > 1.0 else 0
            augmented_labels.append(new_label)
            injected_index = len(augmented_data) - 1
            if new_label == 0:
                injected_normal_indices.append(injected_index)
            else:
                injected_anomaly_indices.append(injected_index)
            # Explainability record: scale_factor here is the label-determining (last-feature) draw.
            point_records.append({
                'index': injected_index,
                'scale': float(scale_factor),
                'local_std': float(np.mean(local_stds)) if local_stds else 0.0,
                'label': int(new_label),
            })
            new_point_counter += 1

    # Convert lists back to numpy arrays with correct shape
    augmented_data = np.array(augmented_data).T  # Transpose to match original data shape
    augmented_labels = np.array(augmented_labels)

    if return_records:
        return augmented_data, augmented_labels, injected_normal_indices, injected_anomaly_indices, point_records
    return augmented_data, augmented_labels, injected_normal_indices, injected_anomaly_indices



def run_off_by_threshold(test_data, trained_models, model_names, dataset, entity, explain=False,
                         metrics=DEFAULT_DECISION_METRICS):
    # Validation: Check if data is too small for off-by-threshold testing
    data = test_data.entities[0].Y
    labels = test_data.entities[0].labels
    
    # Ensure labels are 2D
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)
    
    min_data_size = 100  # Minimum required data points
    data_size = labels.shape[1] if labels.ndim > 1 else labels.shape[0]
    
    if data_size < min_data_size:
        logger.warning(f"Off-by-threshold test skipped: data size {data_size} < minimum {min_data_size}")
        return [], [], [], [], []
    
    # Check if we have both classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning(f"Off-by-threshold test skipped: only one class present in labels (unique values: {unique_labels})")
        return [], [], [], [], []
    
    dataSet_before = copy.deepcopy(test_data)
    factor = .1
    # intersperse_borderline_normal_points expects 1D labels (indexes as labels[i])
    # but labels may have been reshaped to (1, N) above for the uniqueness check — flatten it back
    labels_1d = labels.flatten()
    augmented_data, augmented_labels, injected_normal_indices, injected_anomaly_indices, point_records = \
        intersperse_borderline_normal_points(data, labels_1d, factor, return_records=True)
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
    # One window for the whole stage, taken from the augmented series these
    # detectors are actually scored on, so their VUS values compare.
    want_vus = 'vus' in metrics_required(metrics)
    vus_win = vus_window(test_data.entities[0].Y) if want_vus else None
    for model_name in model_names:
        model = trained_models.get(model_name)
        if not model:
            continue
        try:
            evaluation = evaluate_model(test_data, model, model_name)  # Assume this function returns a dict
        except ScoringTimeout:
            continue
        y_true = evaluation['anomaly_labels'].flatten()
        y_scores = evaluation['entity_scores'].flatten()
        _, _, best_f1, pr_auc, adjusted_y_pred = range_based_precision_recall_f1_auc(y_true, y_scores)
        vus = vus_score(y_scores, y_true, vus_win) if want_vus else float('nan')
        adjusted_y_pred_dict[model_name] = [adjusted_y_pred]
        results[model_name] = [{'f1': best_f1, 'pr_auc': pr_auc, 'vus': vus}]
        logger.info(f"Evaluated {model_name}: F1={best_f1}, PR_AUC={pr_auc}, VUS={vus}")

    ranked_by_f1 = sorted(results.items(), key=lambda x: x[1][0]['f1'], reverse=True)
    ranked_by_f1_names = [item[0] for item in ranked_by_f1]
    ranked_by_pr_auc = sorted(results.items(), key=lambda x: x[1][0]['pr_auc'], reverse=True)
    ranked_by_pr_auc_names = [item[0] for item in ranked_by_pr_auc]
    ranked_by_vus_names = [item[0] for item in sorted(
        results.items(), key=lambda x: rank_key(x[1][0]['vus']), reverse=True)] if want_vus else []

    true_values = np.array(test_data.entities[0].labels).flatten()  # 1 for anomaly, 0 for normal, FLATTEN to 1D
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
    prune_timestamped(directory)

    # plt.show()

    # Explainability (per-point exclusive-win surrogates over the production run; ranking above unchanged)
    if explain:
        try:
            explain_off_by_threshold(point_records, adjusted_y_pred_dict, true_values,
                                     ranked_by_f1_names, model_names, dataset, entity, explain=True)
        except Exception as e:
            logger.error(f"Off-by-threshold explainability failed (non-fatal): {e}")

    return (ranked_by_f1, ranked_by_pr_auc, ranked_by_f1_names,
            ranked_by_pr_auc_names, ranked_by_vus_names)


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
    prune_timestamped(directory)

    # plt.show()


# ════════════════════════════════════════════════════════════════════════════
# Off-by-threshold robustness explainability (explain-only; production ranking
# is unchanged). For the F1 winner of the single production run, a per-competitor
# decision tree explains the winner's *exclusive wins* — the injected borderline
# points the winner classified correctly and that competitor did not — in terms
# of the point's intrinsic properties. Reuses the production run's predictions;
# no extra model evaluations. Gated by `explain` (opt-in via --explain).
# ════════════════════════════════════════════════════════════════════════════

OFFBY_FEATURE_NAMES = ["boundary_distance", "is_anomaly", "local_volatility", "position"]


def build_offby_point_table(point_records, adjusted_y_pred_dict, true_labels,
                            model_names) -> Optional[Dict[str, Any]]:
    """
    Assemble the per-injected-point table from the single production run.

    Features (model-independent point properties), one row per injected point:
      boundary_distance = |scale - 1|, is_anomaly (0/1), local_volatility (local
      std at the site), position (index / N).
    `correct[i, m]` = (model m's production prediction at the point's index ==
    the point's true label). No model inference is run here.

    Returns {X (n×4), feature_names, correct (n×M bool), model_names, indices,
    n_points} or None when there are no injected points / no valid predictions.
    """
    if not point_records:
        return None
    n = len(np.asarray(true_labels).flatten())
    if n == 0:
        return None

    indices = [int(r['index']) for r in point_records]
    X = np.array([[abs(float(r['scale']) - 1.0),
                   float(int(r['label'])),
                   float(r['local_std']),
                   float(r['index']) / float(n)] for r in point_records], dtype=float)

    return ews.join_predictions(
        indices, X, OFFBY_FEATURE_NAMES, adjusted_y_pred_dict, true_labels,
        model_names, stage_label="Off-by explain")


def train_offby_point_surrogates(table, winner, max_depth: int = 3,
                                 random_state: int = 0) -> Dict[str, Any]:
    """
    For the winner, fit one DecisionTreeClassifier per competitor `k` predicting
    the winner's *exclusive wins*: y_i = winner_correct_i AND NOT k_correct_i.

    Per competitor returns export_text rules, feature_importances, train_accuracy,
    n_exclusive_wins, exclusive_win_rate, and the fitted clf. Single-class targets
    (winner never / always strictly beats k) are recorded as degenerate without
    importing sklearn. sklearn is lazy-imported only when a real tree is fit.
    """
    return ews.train_exclusive_win_surrogates(
        table, winner, max_depth=max_depth, random_state=random_state)


# ── Plots ────────────────────────────────────────────────────────────────────

def _offby_explain_dir(dataset, entity) -> str:
    directory = f"myresults/robustness/off_by/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    return directory


def plot_offby_point_tree(info, winner, competitor, dataset, entity, feature_names):
    """One winner-vs-competitor tree; returns its filename, None if degenerate."""
    return ews.plot_exclusive_win_tree(
        info, winner, feature_names,
        directory=_offby_explain_dir(dataset, entity),
        filename=f"{dataset}_{entity}_off_by_point_tree_{winner}_vs_{competitor}.png",
        title=f"Off-by-threshold: where {winner} beats {competitor}\n"
              f"(injected points the winner gets right and {competitor} misses)")


def plot_offby_point_importance(per_competitor, dataset, entity, feature_names) -> None:
    """Bar chart of mean feature importance across all (non-degenerate) competitor trees."""
    ews.plot_exclusive_win_importance(
        per_competitor, feature_names,
        directory=_offby_explain_dir(dataset, entity),
        filename=f"{dataset}_{entity}_off_by_point_importance.png",
        title="Off-by-threshold: which point property most explains the winner's edge")


def explain_off_by_threshold(point_records, adjusted_y_pred_dict, true_labels, ranked_f1_names,
                             model_names, dataset, entity, explain: bool = False) -> Optional[Dict[str, Any]]:
    """
    Off-by-threshold explainability orchestrator (explain-only). Builds the per-point
    table from the production run, picks the F1 winner, fits per-competitor exclusive-win
    surrogates, writes a report + two plots under myresults/robustness/off_by/{ds}/{ent}/,
    and returns the structures. explain=False → None; infeasible table → None.

    The body is `exclusive_win_surrogates.explain_exclusive_win_stage`, shared
    with the GAN stage; only the names and wording below are this stage's own.
    """
    return ews.explain_exclusive_win_stage(
        point_records, adjusted_y_pred_dict, true_labels, ranked_f1_names,
        model_names, dataset, entity, explain,
        stage_label="Off-by-threshold",
        build_table=build_offby_point_table,
        plot_tree_fn=plot_offby_point_tree,
        plot_importance_fn=plot_offby_point_importance,
        explain_dir_fn=_offby_explain_dir,
        tree_prefix="off_by_point_tree_",
        report_stem="off_by_explainability",
        report_heading="Off-by-Threshold Robustness Explainability",
        points_label="Injected borderline points",
        build_ir=ir.build_off_by_ir,
        ir_stem="ir_off_by")

