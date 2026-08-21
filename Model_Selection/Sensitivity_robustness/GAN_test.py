import copy
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow.keras import layers, models

from Metrics.metrics import range_based_precision_recall_f1_auc, prauc, f1_score
from Model_Selection.Sensitivity_robustness.plot_retention import (
    prune_superseded, prune_timestamped)
from Utils.model_selection_utils import evaluate_model


def _ews_module():
    """Import exclusive_win_surrogates.py, tolerating standalone by-path loading of
    this module (e.g. via importlib in test harnesses) where the Model_Selection
    package itself may not be on sys.path — falls back to loading the sibling
    file directly by its own location, the same trick those harnesses use."""
    try:
        from Model_Selection.Sensitivity_robustness import exclusive_win_surrogates as _ews
        return _ews
    except ModuleNotFoundError:
        import importlib.util
        _here = os.path.dirname(os.path.abspath(__file__))
        _spec = importlib.util.spec_from_file_location(
            "exclusive_win_surrogates", os.path.join(_here, "exclusive_win_surrogates.py"))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        return _mod


def _ir_module():
    """Import Explainability.ir with the same standalone-tolerant fallback."""
    try:
        from Explainability import ir as _ir
        return _ir
    except ModuleNotFoundError:
        import importlib.util
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _spec = importlib.util.spec_from_file_location(
            "explainability_ir", os.path.join(_root, "Explainability", "ir.py"))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        return _mod


# Define the generator and discriminator models for GAN
def make_generator_model(input_dim):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(input_dim, activation='tanh')
    ])
    return model


def make_discriminator_model(input_dim):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


# GAN training function with label smoothing and noise
def train_gan(generator, discriminator, data, epochs=100, batch_size=32, noise_dim=10):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    generator_optimizer = tf.keras.optimizers.Adam(0.0001)
    discriminator_optimizer = tf.keras.optimizers.Adam(0.0001)

    # Ensure batch_size doesn't exceed data size
    batch_size = min(batch_size, data.shape[0])
    if batch_size == 0:
        logger.warning("Data size is 0, cannot train GAN")
        return

    for epoch in range(epochs):
        gen_loss = None
        disc_loss = None
        
        num_batches = max(1, data.shape[0] // batch_size)
        for _ in range(num_batches):
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_data = data[idx]

            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            fake_data = generator.predict(noise, verbose=0)

            # Add Gaussian noise to discriminator input
            real_data += 0.1 * np.random.normal(size=real_data.shape)
            fake_data += 0.1 * np.random.normal(size=fake_data.shape)

            with tf.GradientTape() as disc_tape:
                real_output = discriminator(real_data)
                fake_output = discriminator(fake_data)

                # Apply label smoothing
                real_labels = tf.ones_like(real_output) * 0.9
                fake_labels = tf.zeros_like(fake_output) + 0.1

                disc_loss_real = cross_entropy(real_labels, real_output)
                disc_loss_fake = cross_entropy(fake_labels, fake_output)
                disc_loss = disc_loss_real + disc_loss_fake

            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            with tf.GradientTape() as gen_tape:
                fake_data = generator(noise, training=True)
                fake_output = discriminator(fake_data)
                gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        if epoch % 50 == 0 and gen_loss is not None and disc_loss is not None:
            logger.info(f"Epoch {epoch}: Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}")


# Function to generate new borderline points
def generate_borderline_points(generator, num_samples=100, noise_dim=10):
    noise = np.random.normal(0, 1, (num_samples, noise_dim))
    generated_data = generator.predict(noise, verbose=0)
    return generated_data


# The candidate pool is this multiple of the injection budget (paper Eq. 7-8:
# generate K candidates, keep the B most ambiguous). A plain 10x oversample —
# `generator.predict` on a few thousand rows costs nothing beside 100 epochs of
# adversarial training.
GAN_CANDIDATE_OVERSAMPLE = 10


def _to_tanh_space(x):
    """Series space -> the generator's tanh output range.

    Paper Sec. V-B-1, "Data preparation": inputs are linearly scaled to [-1, 1]
    to match the generator's tanh output layer. `Datasets/load.py` MinMax-scales
    every series to [0, 1], so without this half the generator's range falls
    outside the data's own range and the discriminator can separate real from
    fake on magnitude alone — the generated points are then trivially detectable
    rather than borderline, which is the property the whole test rests on.

    A fixed affine map rather than a refitted scaler: it inverts exactly, carries
    no state, and cannot leak (the test split is already scaled by the scaler
    fitted on the training split upstream).
    """
    return 2.0 * np.asarray(x, dtype=float) - 1.0


def _from_tanh_space(x):
    """Inverse of `_to_tanh_space` — back to the series' own scale."""
    return (np.asarray(x, dtype=float) + 1.0) / 2.0


def integrate_gan_with_dataset(data, labels, factor=0.1, return_records=False):
    """
    Integrates generated GAN data into the existing dataset with dynamic labeling,
    and returns the updated dataset, labels, indices of normal and anomaly injected points,
    and the total count of labels after integration.

    Args:
        data (np.ndarray): Original data array of shape (n_features, n_samples).
        labels (np.ndarray): Original label array.
        factor (float): Injection budget rho as a fraction of the original sample
            count (paper default ~0.1).
        return_records (bool): Explainability opt-in. When True, additionally return a
            list of per-injected-point records as a 6th value. The records are built
            from values the injection branch has already computed plus read-only
            statistics of ``data``, so no extra RNG draws are made and the first five
            return values — and therefore the production ranking — are byte-for-byte
            identical to the default ``return_records=False`` path.

    Returns:
        tuple: Updated data, labels, indices of normal and anomaly injected points,
               and the total number of labels (plus the per-point records when
               ``return_records=True``).
    """
    input_dim = data.shape[0]  # Assuming data is of shape (n_features, n_samples)
    generator = make_generator_model(input_dim)
    discriminator = make_discriminator_model(input_dim)
    point_records = []

    # Filter to only normal (non-anomaly) data for GAN training
    # This ensures the GAN learns P(normal) and generates borderline points
    # near the normal distribution boundary, avoiding leakage from anomaly patterns
    #
    # `labels` arrives 2-D (1, n) from Entity, and np.where on a 2-D array returns
    # (row_indices, col_indices) — taking [0] took the ROWS, an array of zeros, so
    # clean_data was n_normal copies of column 0 and the GAN trained on a single
    # repeated point. Flatten first so these are the timestamps they claim to be.
    labels_1d = np.asarray(labels).flatten()
    normal_indices = np.where(labels_1d == 0)[0]
    if len(normal_indices) > 0:
        clean_data = data[:, normal_indices]
    else:
        # Fallback: if no normal points labeled, use all data
        logger.warning("No normal points found (all labels are 1), using all data for GAN training")
        clean_data = data

    # Train the GAN on clean, non-anomalous data only
    train_gan(generator, discriminator, _to_tanh_space(clean_data.T), epochs=100,
              batch_size=32, noise_dim=input_dim)

    # Injection budget B (paper: rho ~ 10% of the original samples). `labels` is
    # 2-D here, so the previous `len(labels)` was 1 rather than n_samples — which
    # is why this stage injected exactly one point per run regardless of series
    # length, while the window loop below was already sized for one per window.
    budget = int(factor * labels_1d.shape[0])
    if budget == 0:
        budget = 1  # Ensure at least one sample

    # Paper Eq. 7-8: generate a candidate pool, measure each candidate's ambiguity
    # against the discriminator's decision threshold, and keep the B most ambiguous
    # — the points sitting where "normal" and "anomalous" are hardest to separate.
    # Without this step every generated point is injected, and nothing makes the
    # injected set borderline.
    candidates = generate_borderline_points(
        generator, num_samples=GAN_CANDIDATE_OVERSAMPLE * budget, noise_dim=input_dim)
    candidate_scores = discriminator.predict(candidates, verbose=0).flatten()
    tau = float(np.mean(candidate_scores))          # the decision threshold
    candidate_ambiguity = np.abs(candidate_scores - tau)                    # Eq. 7
    keep = np.argsort(candidate_ambiguity, kind="stable")[:budget]          # Eq. 8

    # Scores stay in the space the discriminator was trained on; the points
    # themselves go back to the series' scale before they are injected into it.
    borderline_points = _from_tanh_space(candidates[keep])
    discriminator_outputs = candidate_scores[keep]
    ambiguity = candidate_ambiguity[keep]

    # Use the discriminator to dynamically label the generated points, against the
    # same threshold the selection used: paper Eq. 9, y_hat(x) = 1[D(x) >= tau],
    # yielding both near-normal (0) and near-anomalous (1) behaviours.
    new_labels = np.where(discriminator_outputs > tau, 1, 0)
    logger.info(f"GAN: kept {len(keep)} of {len(candidates)} candidates, "
                f"tau={tau:.4f}, ambiguity<={float(ambiguity.max()):.4f}, "
                f"{int(new_labels.sum())} labelled anomalous")

    # Integrate the generated points into the original dataset using windows
    num_windows = max(1, len(data[0]) // 10)  # Divide the data into windows, ensure at least 1
    indices_to_insert = np.array_split(np.arange(len(data[0])), num_windows)

    # Half-width of the neighbourhood the explainability features describe the
    # injection site with. Off-by's formula (its `contextual_length`), floored so
    # a standard deviation is never taken over three points.
    w_ctx = max(5, int(0.05 * factor * data.shape[1]))

    integrated_data = []
    integrated_labels = []
    injected_normal_indices = []
    injected_anomaly_indices = []

    current_index = 0
    for window in indices_to_insert:
        # Add the original data points for this window
        integrated_data.append(data[:, window])

        # Handle labels consistently - always flatten to 1D for concatenation
        if labels.ndim == 1:
            window_labels = labels[window]
        else:
            # If 2D (1, n), extract the row and index columns
            window_labels = labels.flatten()[window]
        integrated_labels.append(window_labels)
        current_index += len(window)

        # Add a generated point in the middle of the window
        if len(borderline_points) > 0:
            integrated_data.append(borderline_points[:1].T)
            # Add single label as scalar (will be concatenated properly)
            integrated_labels.append(new_labels[:1])
            if new_labels[0] == 0:
                injected_normal_indices.append(current_index)
            else:
                injected_anomaly_indices.append(current_index)
            if return_records:
                # Explainability record. Every value here is either already in hand
                # (the point, its discriminator score, its ambiguity, its label) or a
                # read-only statistic of `data` — no RNG is touched, so the four
                # production return values are unaffected.
                point = borderline_points[0]
                site = int(window[-1])
                ctx = data[:, max(0, site - w_ctx):min(data.shape[1], site + w_ctx + 1)]
                local_mean = np.mean(ctx, axis=1)
                local_std = np.std(ctx, axis=1)
                point_records.append({
                    'index': current_index,
                    'disc_score': float(discriminator_outputs[0]),
                    'tau': tau,
                    'ambiguity': float(ambiguity[0]),
                    'label': int(new_labels[0]),
                    'magnitude': float(np.mean(np.abs(point))),
                    'spread': float(np.std(point)),
                    'context_gap': float(np.mean(np.abs(point - local_mean))),
                    'local_std': float(np.mean(local_std)),
                })
            current_index += 1
            borderline_points = borderline_points[1:]
            new_labels = new_labels[1:]
            discriminator_outputs = discriminator_outputs[1:]
            ambiguity = ambiguity[1:]

    integrated_data = np.concatenate(integrated_data, axis=1)
    # Concatenate all label arrays (all are now 1D)
    integrated_labels = np.concatenate(integrated_labels, axis=0)

    # Count the total number of labels after integration
    total_labels_count = len(integrated_labels)

    if return_records:
        return integrated_data, integrated_labels, np.array(injected_normal_indices), np.array(
            injected_anomaly_indices), total_labels_count, point_records
    return integrated_data, integrated_labels, np.array(injected_normal_indices), np.array(
        injected_anomaly_indices), total_labels_count





def run_Gan(test_data, trained_models, model_names, dataset, entity, explain=False):
    # Validation: Check if data is too small for GAN testing
    data = test_data.entities[0].Y
    labels = test_data.entities[0].labels
    
    # Ensure labels are 2D
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)
    
    min_data_size = 100  # Minimum required data points for GAN testing
    data_size = labels.shape[1] if labels.ndim > 1 else labels.shape[0]
    
    if data_size < min_data_size:
        logger.warning(f"GAN test skipped: data size {data_size} < minimum {min_data_size}")
        return [], [], [], []
    
    # Check if we have both classes (anomalies and normal points)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning(f"GAN test skipped: only one class present in labels (unique values: {unique_labels})")
        return [], [], [], []
    
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataSet_before = copy.deepcopy(test_data)
    factor = .1
    augmented_data, augmented_labels, injected_normal_indices, injected_anomaly_indices, total_labels_count, point_records = \
        integrate_gan_with_dataset(data, labels, factor=factor, return_records=True)
    test_data.entities[0].Y = np.array(augmented_data)
    test_data.entities[0].labels = np.array(augmented_labels)
    n_times = test_data.entities[0].n_time
    test_data.total_time = total_labels_count
    test_data.entities[0].n_time = total_labels_count
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

    # Filter out models with no results before sorting
    valid_results = {k: v for k, v in results.items() if len(v) > 0}
    
    if not valid_results:
        logger.warning("No valid GAN test results found, skipping ranking")
        return [], [], [], []
    
    ranked_by_f1 = sorted(valid_results.items(), key=lambda x: x[1][0]['f1'], reverse=True)
    ranked_by_f1_names = [item[0] for item in ranked_by_f1]
    ranked_by_pr_auc = sorted(valid_results.items(), key=lambda x: x[1][0]['pr_auc'], reverse=True)
    ranked_by_pr_auc_names = [item[0] for item in ranked_by_pr_auc]

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
        f'True vs. Predicted Anomalies \n Misclassified Labels: {misclassified_count}\n Total Anomalies: {total_anomalies} \n Total Data: {total_data}')
    plt.xlabel('Index')
    plt.ylabel('Anomaly Presence')
    plt.yticks([0, 1], ['No Anomaly', 'Anomaly'])  # Set y-ticks to be explicit about what 0 and 1 represent
    plt.legend()
    plt.grid(True)

    # Specify the directory
    directory = f'myresults/robustness/GAN/{dataset}/{entity}/'
    filename = f'{dataset}_{entity}_Misclassified_Anomalies_{date_time_string}_.png'
    full_path = os.path.join(directory, filename)

    # Check if the directory exists, and if not, create it
    os.makedirs(directory, exist_ok=True)  # Add exist_ok=True to avoid FileExistsError

    # Save the figure
    plt.savefig(full_path, dpi=300)  # Save as PNG file with high resolution
    prune_timestamped(directory)

    # plt.show()

    # Explainability (per-point exclusive-win surrogates over the production run; ranking above unchanged)
    if explain:
        try:
            explain_gan_robustness(point_records, adjusted_y_pred_dict, true_values,
                                   ranked_by_f1_names, model_names, dataset, entity, explain=True)
        except Exception as e:
            logger.error(f"GAN robustness explainability failed (non-fatal): {e}")

    return ranked_by_f1, ranked_by_pr_auc, ranked_by_f1_names, ranked_by_pr_auc_names

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
    if injected_normal_indices.size > 0:
        axes[1].scatter(injected_normal_indices,
                        [augmented_data[feature_index, idx] for idx in injected_normal_indices],
                        color='green', label='Injected Normal Points', marker='o', s=50)

    # Highlight injected anomaly points in red
    if injected_anomaly_indices.size > 0:
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
    directory = f'myresults/robustness/GAN/{dataset}/{entity}/'
    filename = f'{dataset}_{entity}_Data_vs_DataWithAnomalies_{date_time_string}.png'
    full_path = os.path.join(directory, filename)

    # Check if the directory exists, and if not, create it
    os.makedirs(directory, exist_ok=True)  # Add exist_ok=True to avoid FileExistsError

    # Save the figure
    plt.savefig(full_path, dpi=300)  # Save as PNG file with high resolution
    prune_timestamped(directory)

    # plt.show()


# ════════════════════════════════════════════════════════════════════════════
# GAN robustness explainability (explain-only; production ranking is unchanged).
# For the F1 winner of the single production run, a per-competitor decision tree
# explains the winner's *exclusive wins* — the injected GAN points the winner
# classified correctly and that competitor did not — in terms of meta-features of
# the generated signal and of the site it was injected into. Reuses the production
# run's predictions; no extra model evaluations. Gated by `explain` (opt-in via
# --explain). The surrogate machinery itself is shared with off-by-threshold; see
# exclusive_win_surrogates.py.
# ════════════════════════════════════════════════════════════════════════════

# The generator is G: R^d -> R^d with a tanh output, so a generated signal is ONE
# multivariate timestep, not a window — within-signal temporal statistics
# (autocorrelation, spectral energy) are undefined on a length-1 series. The
# temporal content therefore comes from the injection site: `context_gap` and
# `local_volatility`.
#
# The raw discriminator score is deliberately NOT a feature: `ambiguity` and
# `is_anomaly` jointly determine it, so a third column would be collinear and
# would split tree importance across redundant copies. This mirrors off-by, which
# carries |scale - 1| and scale > 1 but never the raw scale.
GAN_FEATURE_NAMES = ["ambiguity", "is_anomaly", "signal_magnitude", "signal_spread",
                     "context_gap", "local_volatility", "position"]


def build_gan_point_table(point_records, adjusted_y_pred_dict, true_labels,
                          model_names) -> Optional[Dict[str, Any]]:
    """
    Assemble the per-injected-point table from the single production run.

    Features (model-independent properties of the generated point), one row per
    injected point: ambiguity (|D(x) - tau|, paper Eq. 7), is_anomaly (the Eq. 9
    label), signal_magnitude and signal_spread across channels, context_gap and
    local_volatility at the injection site, and position (index / N).
    `correct[i, m]` = (model m's production prediction at the point's index ==
    the point's label). No model inference is run here.

    Returns {X (n×7), feature_names, correct (n×M bool), model_names, indices,
    n_points} or None when there are no injected points / no valid predictions.
    """
    if not point_records:
        return None
    n = len(np.asarray(true_labels).flatten())
    if n == 0:
        return None

    indices = [int(r['index']) for r in point_records]
    X = np.array([[float(r['ambiguity']),
                   float(int(r['label'])),
                   float(r['magnitude']),
                   float(r['spread']),
                   float(r['context_gap']),
                   float(r['local_std']),
                   float(r['index']) / float(n)] for r in point_records], dtype=float)

    return _ews_module().join_predictions(
        indices, X, GAN_FEATURE_NAMES, adjusted_y_pred_dict, true_labels,
        model_names, stage_label="GAN explain")


def train_gan_point_surrogates(table, winner, max_depth: int = 3,
                               random_state: int = 0) -> Dict[str, Any]:
    """
    For the winner, fit one DecisionTreeClassifier per competitor `k` predicting
    the winner's *exclusive wins*: y_i = winner_correct_i AND NOT k_correct_i.
    """
    return _ews_module().train_exclusive_win_surrogates(
        table, winner, max_depth=max_depth, random_state=random_state)


# ── Plots ────────────────────────────────────────────────────────────────────

def _gan_explain_dir(dataset, entity) -> str:
    directory = f"myresults/robustness/GAN/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    return directory


def plot_gan_point_tree(info, winner, competitor, dataset, entity, feature_names):
    """One winner-vs-competitor tree; returns its filename, None if degenerate."""
    return _ews_module().plot_exclusive_win_tree(
        info, winner, feature_names,
        directory=_gan_explain_dir(dataset, entity),
        filename=f"{dataset}_{entity}_gan_point_tree_{winner}_vs_{competitor}.png",
        title=f"GAN perturbations: where {winner} beats {competitor}\n"
              f"(injected points the winner gets right and {competitor} misses)")


def plot_gan_point_importance(per_competitor, dataset, entity, feature_names) -> None:
    """Bar chart of mean feature importance across all (non-degenerate) competitor trees."""
    _ews_module().plot_exclusive_win_importance(
        per_competitor, feature_names,
        directory=_gan_explain_dir(dataset, entity),
        filename=f"{dataset}_{entity}_gan_point_importance.png",
        title="GAN perturbations: which point property most explains the winner's edge")


def explain_gan_robustness(point_records, adjusted_y_pred_dict, true_labels, ranked_f1_names,
                           model_names, dataset, entity, explain: bool = False) -> Optional[Dict[str, Any]]:
    """
    GAN robustness explainability orchestrator (explain-only). Builds the per-point
    table from the production run, picks the F1 winner, fits per-competitor exclusive-win
    surrogates, writes a report + two plots under myresults/robustness/GAN/{ds}/{ent}/,
    and returns the structures. explain=False → None; infeasible table → None.
    """
    if not explain:
        return None
    table = build_gan_point_table(point_records, adjusted_y_pred_dict, true_labels, model_names)
    if table is None:
        logger.warning("GAN explainability skipped: no injected points / valid predictions.")
        return None

    models = table["model_names"]
    # Winner = highest-ranked F1 model that actually has predictions; else first valid.
    winner = next((m for m in (ranked_f1_names or []) if m in models), models[0])
    ranked_valid = [m for m in (ranked_f1_names or []) if m in models]
    runnerup = next((m for m in ranked_valid if m != winner), None)

    surrogate_note = ""
    res = {"feasible": False, "winner": winner, "feature_names": table["feature_names"], "per_competitor": {}}
    try:
        res = train_gan_point_surrogates(table, winner)
    except ImportError:
        surrogate_note = "scikit-learn unavailable — per-competitor surrogates skipped."
        logger.warning(f"GAN explainability: {surrogate_note}")

    per_competitor = res.get("per_competitor", {})
    if not surrogate_note:
        # Plot every generated surrogate tree, one per competitor (degenerate ones,
        # whose clf is None, are skipped inside plot_gan_point_tree).
        written = [plot_gan_point_tree(info, winner, k, dataset, entity,
                                        table["feature_names"])
                   for k, info in per_competitor.items()]
        plot_gan_point_importance(per_competitor, dataset, entity, table["feature_names"])
        # Whatever an earlier run left here describes a different outcome —
        # a different winner, or the same winner against differently-spelled
        # competitors — and the picker cannot tell the two apart. Pruned
        # AFTER the new set is on disk, so a run that dies mid-plot leaves
        # the previous set rather than deleting it and not replacing it.
        prune_superseded(_gan_explain_dir(dataset, entity),
                         f"{dataset}_{entity}_gan_point_tree_",
                         [n for n in written if n])

    directory = _gan_explain_dir(dataset, entity)
    report_path = os.path.join(directory, f"{dataset}_{entity}_gan_explainability.txt")
    with open(report_path, "w") as f:
        f.write("=== GAN Robustness Explainability ===\n")
        f.write(f"Dataset: {dataset}  |  Entity: {entity}\n")
        f.write(f"Models with predictions ({len(models)}): {', '.join(models)}\n")
        f.write(f"Injected GAN points: {table['n_points']}\n")
        f.write(f"Features: {', '.join(table['feature_names'])}\n")
        f.write(f"F1 winner (production ranking): {winner}\n")
        f.write("(Explains the actual production run; correctness is F1/prediction-side — "
                "PR-AUC has no per-point correct/incorrect. The production ranking is unchanged.)\n\n")

        if surrogate_note:
            f.write(surrogate_note + "\n")
        elif not per_competitor:
            f.write("No competitors to compare against (winner is the only model with predictions).\n")
        else:
            order = [m for m in ranked_valid if m in per_competitor] + \
                    [m for m in per_competitor if m not in ranked_valid]
            agg: Dict[str, List[float]] = {fn: [] for fn in table["feature_names"]}
            for k in order:
                info = per_competitor[k]
                f.write(f"--- {winner} vs {k} ---\n")
                f.write(f"Exclusive wins: {info['n_exclusive_wins']} "
                        f"({info['exclusive_win_rate']:.2%} of injected points)\n")
                if info.get("degenerate"):
                    f.write(f"    {info['rules_text']}\n\n")
                    continue
                f.write(f"Surrogate train accuracy (in-sample fit): {info['train_accuracy']:.3f}\n")
                cv_acc = info.get("cv_accuracy", float('nan'))
                if not np.isnan(cv_acc):
                    f.write(f"Surrogate held-out accuracy ({info.get('cv_method', 'cv')}, "
                            f"{info.get('cv_accuracy_std', float('nan')):.3f} std): {cv_acc:.3f}\n")
                elif info.get("cv_note"):
                    f.write(f"Surrogate held-out accuracy: not estimated ({info['cv_note']})\n")
                imps = sorted(info["feature_importances"].items(), key=lambda kv: kv[1], reverse=True)
                f.write("Feature importances: "
                        + ", ".join(f"{fn} {im:.2f}" for fn, im in imps if im > 0) + "\n")
                for fn, im in info["feature_importances"].items():
                    agg[fn].append(im)
                f.write("Rules (1 = point the winner uniquely gets right):\n")
                for line in info["rules_text"].rstrip().splitlines():
                    f.write(f"    {line}\n")
                f.write("\n")
            mean_imp = {fn: (float(np.mean(v)) if v else 0.0) for fn, v in agg.items()}
            if any(mean_imp.values()):
                top = max(mean_imp.items(), key=lambda kv: kv[1])
                f.write(f"Across competitors, the winner's edge is best explained by: "
                        f"{top[0]} (mean importance {top[1]:.2f}).\n")

    result = {"table": table, "winner": winner, "runnerup": runnerup,
              "surrogates": res, "n_points": table["n_points"]}

    # ── Intermediate Representation (grounded LLM input; non-fatal) ─────────
    try:
        _ir = _ir_module()
        _ir.write_stage_ir(
            _ir.build_gan_ir(dataset, entity, result, ranked_f1_names),
            dataset, entity, "ir_gan")
    except Exception as e:
        logger.error(f"GAN IR emission failed (non-fatal): {e}")

    return result