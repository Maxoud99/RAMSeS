import copy
import os
import random
import traceback
from typing import List, Dict, Any, Optional
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
                 epsilon: float, context: np.ndarray) -> Tuple[str, bool]:
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
    - Tuple[str, bool]: (chosen model name, was_random) where was_random is True iff the
      epsilon-greedy random branch fired. The flag enables downstream classification of
      the selection as random / exploitation / informed_exploration.
    """
    if random.random() < epsilon:
        chosen_model = random.choice(list(models.keys()))
        logger.info(f"Epsilon-Greedy: Randomly chosen model {chosen_model}")
        return chosen_model, True

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
    return chosen_model, False


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


def compute_expected_rewards(means: Dict[str, np.ndarray], context: np.ndarray) -> Dict[str, float]:
    """
    Compute the expected reward for every model given current posterior means and a context vector.

    E[reward | model_k, context_t] = mu_k^T * context_t

    Parameters
    ----------
    means : Dict[str, np.ndarray]
        Posterior mean vectors mu_k. Each may be 1-D (d,) or column (d, 1).
    context : np.ndarray
        The flattened data window at the current timestep, shape (d,).

    Returns
    -------
    Dict[str, float]
        Keys are model names; values are scalar expected rewards mu_k^T * x.
        Values can be negative (standardised data, uninitialised means).
    """
    return {m: float(np.dot(mu.flatten(), context.flatten())) for m, mu in means.items()}


def classify_selection(
    chosen_model: str,
    was_random: bool,
    expected_rewards: Dict[str, float],
) -> str:
    """
    Categorize a model selection into one of three behavioral states.

    States
    ------
    - "random"               : ε-greedy random pick fired (exploration floor).
    - "exploitation"         : chosen model equals argmax_k (mu_k^T * x) over current
                               (pre-update) posterior means; the agent picked what it
                               already believed was best.
    - "informed_exploration" : chosen via Thompson sampling but differs from the mean-
                               based argmax; posterior uncertainty steered the agent
                               away from its mean-best guess.

    Parameters
    ----------
    chosen_model : str
        The model that was actually selected at this window.
    was_random : bool
        True iff ε-greedy fired. Takes precedence over the argmax comparison so that
        an ε-greedy pick that happens to match the argmax is still labeled "random".
    expected_rewards : Dict[str, float]
        Output of compute_expected_rewards(means, context) using the PRE-update means
        (the beliefs that informed this decision).

    Returns
    -------
    str : one of {"random", "exploitation", "informed_exploration"}.
    """
    if was_random:
        return "random"
    expected_best = max(expected_rewards, key=expected_rewards.get)
    return "exploitation" if chosen_model == expected_best else "informed_exploration"


def compute_shap_values(mean: np.ndarray, context: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """
    Per-feature SHAP attribution for a linear model E[R] = mean^T x.

    Closed form for linear models (matches shap.LinearExplainer with
    feature_dependence='independent'):

        phi_0 = mean^T baseline          (baseline expected reward)
        phi_i = mean_i * (x_i - baseline_i)
        phi_0 + sum(phi) = mean^T x      (additivity guarantee)

    Parameters
    ----------
    mean : np.ndarray
        Linear weights (the posterior mean mu_k). May be 1-D (d,) or column (d, 1).
    context : np.ndarray
        The instance to explain, shape (d,).
    baseline : np.ndarray
        The reference distribution mean E[X], shape (d,).

    Returns
    -------
    np.ndarray
        Per-feature SHAP attributions, shape (d,).
    """
    return mean.flatten() * (context.flatten() - baseline.flatten())


def aggregate_shap_per_channel(shap_values: np.ndarray, n_channels: int) -> np.ndarray:
    """
    Sum SHAP values within each channel's window-of-timesteps slice.

    The flattened feature vector is assumed to be reshape(n_channels, window_size)
    flattened in C-order (numpy default), so feature i belongs to channel i // window_size.

    If shap_values.size is not divisible by n_channels (should not happen after the
    n_features fix), trailing entries are dropped.

    Parameters
    ----------
    shap_values : np.ndarray
        Per-feature SHAP values, shape (d,).
    n_channels : int
        Number of sensor channels.

    Returns
    -------
    np.ndarray
        Per-channel contributions, shape (n_channels,).
    """
    if n_channels <= 0:
        return np.zeros(0)
    window_size = shap_values.size // n_channels
    if window_size == 0:
        return np.zeros(n_channels)
    return shap_values[: n_channels * window_size].reshape(n_channels, window_size).sum(axis=1)


def detect_regime_shifts(
    expected_rewards_history: Dict[str, List[float]],
    smoothing_window: int = 5,
    min_regime_length: int = 3,
) -> Tuple[List[Dict], List[str]]:
    """
    Detect sustained changes in the dominant model from expected-reward history.

    A regime is a sustained period where one model holds the highest expected reward.
    A regime shift is recorded when the new dominant model persists for at least
    min_regime_length consecutive windows. Shorter changes are classified as blips.

    Parameters
    ----------
    expected_rewards_history : Dict[str, List[float]]
        Per-model expected reward sequences. NaN values (from skipped windows) are handled.
    smoothing_window : int, default 5
        Width of rolling mean used to suppress per-window noise (1 = no smoothing).
    min_regime_length : int, default 3
        Minimum consecutive windows a model must dominate to constitute a true regime.

    Returns
    -------
    regime_shifts : List[Dict]
        Each dict has keys: window, from_model, to_model, reward_delta, regime_length.
        reward_delta is smoothed[to_model] - smoothed[from_model] at the shift window.
        regime_length is the duration (windows) of the OLD regime.
    blip_windows : List[str]
        Human-readable labels for transient dominance changes below min_regime_length.
    """
    model_list = list(expected_rewards_history.keys())
    if not model_list:
        return [], []

    T = len(expected_rewards_history[model_list[0]])
    if T == 0:
        return [], []

    reward_matrix = np.array([expected_rewards_history[m] for m in model_list], dtype=float)

    # Rolling mean smoothing per model; replace NaN with 0 for convolution stability
    nan_mask = np.isnan(reward_matrix)
    safe_matrix = np.where(nan_mask, 0.0, reward_matrix)
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed = np.array([np.convolve(safe_matrix[k], kernel, 'same') for k in range(len(model_list))])
    else:
        smoothed = safe_matrix.copy()

    # Dominant model per window (None where all models have NaN)
    all_nan = np.all(nan_mask, axis=0)
    dominant = [
        None if all_nan[t] else model_list[int(np.argmax(smoothed[:, t]))]
        for t in range(T)
    ]

    # Run-length encode the non-None dominant sequence into segments
    segments = []
    i = 0
    while i < T:
        if dominant[i] is None:
            i += 1
            continue
        j = i + 1
        while j < T and dominant[j] == dominant[i]:
            j += 1
        segments.append((dominant[i], i, j - i))
        i = j

    if len(segments) <= 1:
        return [], []

    # Walk segments: sustained change → regime shift; short change → blip
    regime_shifts = []
    blip_windows = []
    prev_model = segments[0][0]
    prev_start = segments[0][1]

    for model, start, length in segments[1:]:
        if model == prev_model:
            # Continuation of current regime after a blip sequence
            continue
        if length >= min_regime_length:
            k_new = model_list.index(model)
            k_old = model_list.index(prev_model)
            regime_shifts.append({
                'window': start,
                'from_model': prev_model,
                'to_model': model,
                'reward_delta': float(smoothed[k_new, start] - smoothed[k_old, start]),
                'regime_length': start - prev_start,
            })
            prev_model = model
            prev_start = start
        else:
            for t in range(start, start + length):
                blip_windows.append(f"window {t}: {prev_model} -> {model} (blip)")

    return regime_shifts, blip_windows


def reconstruct_regime_segments(
    regime_shifts: List[Dict],
    T: int,
    fallback_model: str = 'N/A',
) -> List[Tuple[int, int, str, int]]:
    """
    Rebuild contiguous regime segments from the list of regime-shift events.

    Parameters
    ----------
    regime_shifts : List[Dict]
        Output of detect_regime_shifts(); each dict has 'window', 'from_model',
        'to_model'.
    T : int
        Total number of windows.
    fallback_model : str
        Model name to use for the single segment when there are no shifts.

    Returns
    -------
    List[Tuple[int, int, str, int]]
        One tuple per regime: (start, end_inclusive, dominant_model, duration).
    """
    if T <= 0:
        return []
    if regime_shifts:
        starts = [0] + [s['window'] for s in regime_shifts]
        ends = [s['window'] for s in regime_shifts] + [T]
        models_seq = [regime_shifts[0]['from_model']] + [s['to_model'] for s in regime_shifts]
        return [(rs, re - 1, rm, re - rs) for rm, rs, re in zip(models_seq, starts, ends)]
    return [(0, T - 1, fallback_model, T)]


def fit_linear_thompson_sampling(dataset,
                                 models: Dict[str, Any], data: np.ndarray, targets: np.ndarray,
                                 initial_epsilon: float = 0.2,
                                 epsilon_decay: float = 0.99, f1_weight: float = 0.7, pr_auc_weight: float = 0.3,
                                 iterations: int = 100,
                                 explain: bool = False) -> Tuple[
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
    _exp_rewards_hist: Dict[str, List[float]] = {m: [] for m in models}
    _pre_exp_rewards_hist: Dict[str, List[float]] = {m: [] for m in models}
    _l2_norm_hist: Dict[str, List[float]] = {m: [] for m in models}
    _selection_states: List[str] = []
    _window_contexts: List[np.ndarray] = []

    for iteration in range(num_windows):
        logger.info(f"Iteration {iteration + 1}")
        try:
            # Pass the current window as context so selection uses theta_tilde^T * x.
            # Normalise to unit length so that datasets with large sensor values
            # (e.g. SMD with 38 channels) do not cause xxᵀ to explode and collapse Σ.
            context = data_windows[iteration].flatten()
            context = context / (np.linalg.norm(context) + 1e-10)
            chosen_model_name, was_random = sample_model(models, means, covariances, epsilon, context)
        except ValueError as e:
            logger.error(f"Error sampling model: {e}")
            continue  # Skip to the next iteration on error
        chosen_model = models[chosen_model_name]
        list_of_chosen_models.append(chosen_model_name)

        if explain:
            # Expected rewards from the PRE-update means — the beliefs at the time
            # of sampling. Used for classification and for the report's Dominant /
            # Top E[Reward] columns (which describe the decision, not its aftermath).
            pre_update_rewards = compute_expected_rewards(means, context)
            _selection_states.append(classify_selection(chosen_model_name, was_random, pre_update_rewards))
            for _m in models:
                _pre_exp_rewards_hist[_m].append(pre_update_rewards[_m])
            # Store the (L2-normalised) context so SHAP can be computed per window later.
            _window_contexts.append(context)

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
            # to context above so that θ̃ᵀx (selection) and the posterior update operate
            # in the same feature space.
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

            if explain:
                for _m, _mu in means.items():
                    _exp_rewards_hist[_m].append(float(np.dot(_mu.flatten(), context)))
                    _l2_norm_hist[_m].append(float(np.dot(_mu.flatten(), _mu.flatten())))

            logger.info(
                f"Window {iteration + 1}: Model {chosen_model_name} - F1 Score = {f1}, PR AUC = {pr_auc}, Reward = {reward}")
            logger.info(f"Means: {means}")
            logger.info(f"Covariances: {covariances}")

        except Exception as e:
            logger.error(f"Error evaluating model {chosen_model_name}: {e}")
            detailed_traceback = traceback.format_exc()
            print(detailed_traceback)
            if explain:
                for _m in models:
                    _exp_rewards_hist[_m].append(float('nan'))
                    _l2_norm_hist[_m].append(float('nan'))
            continue  # Skip the current iteration on error

        epsilon *= epsilon_decay

        history.append({model_name: means[model_name].flatten() for model_name in models})
        logger.info(f"Finished iteration {iteration + 1}")

    if explain:
        # Build the SHAP payload from the per-window contexts the bandit actually
        # processed (_window_contexts), which stays index-aligned with the regime
        # detection and the explainability histories. Each context is already
        # L2-normalised. all_contexts enables per-window / per-regime SHAP plots.
        n_channels = data_windows[0].shape[0] if data_windows else 0
        if _window_contexts:
            baseline_context = np.mean(_window_contexts, axis=0)
            explanation_context = _window_contexts[-1]
        else:
            baseline_context = np.zeros(n_features)
            explanation_context = np.zeros(n_features)
        shap_payload = {
            "explanation_context": explanation_context,
            "baseline_context": baseline_context,
            "all_contexts": _window_contexts,
            "n_channels": n_channels,
        }
        return (means, covariances, history, list_of_chosen_models,
                _exp_rewards_hist, _l2_norm_hist, _selection_states,
                _pre_exp_rewards_hist, shap_payload)
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


def plot_expected_rewards(
    expected_rewards_history: Dict[str, List[float]],
    regime_shifts: List[Dict],
    model_names: List[str],
    dataset: str,
    entity: str,
    iterations: int,
    smooth: bool = False,
) -> None:
    """
    Plot expected reward evolution for all models with regimes annotated.

    Regime regions are shaded by dominant model, regime boundaries are marked
    with dashed vertical lines, and every regime is labelled at its centre
    (vertically) with the model that dominates it.

    smooth : bool
        When False (default) each trajectory is the raw per-window value, saved
        as expected_rewards_{iterations}.png. When True each trajectory is
        Gaussian-smoothed (sigma=2) and saved as
        expected_rewards_smoothed_{iterations}.png. Both variants are produced
        per run so the raw and smoothed views can be compared side by side.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })

    T = len(next(iter(expected_rewards_history.values()))) if expected_rewards_history else 0
    colour_map = {name: plt.cm.tab20(i / max(len(model_names), 1)) for i, name in enumerate(model_names)}

    fig, ax = plt.subplots(figsize=(10, 5))

    for model_name in model_names:
        raw = np.array(expected_rewards_history.get(model_name, []), dtype=float)
        if raw.size == 0:
            continue
        nan_mean = float(np.nanmean(raw)) if not np.all(np.isnan(raw)) else 0.0
        nan_free = np.where(np.isnan(raw), nan_mean, raw)
        series = gaussian_filter1d(nan_free, sigma=2) if smooth else nan_free
        ax.plot(range(T), series, label=model_name, linewidth=1.4, color=colour_map[model_name])

    # Shade each regime, mark every boundary, and label every regime with the
    # model that dominates it — written vertically, centred in the regime span.
    if T > 0 and model_names:
        segments = reconstruct_regime_segments(regime_shifts, T, fallback_model=model_names[0])
        y_top = ax.get_ylim()[1]
        for (start, end, rm, _duration) in segments:
            if rm in colour_map:
                ax.axvspan(start, end + 1, alpha=0.08, color=colour_map[rm], lw=0)
            center = (start + end + 1) / 2.0
            ax.text(center, y_top * 0.97, rm, fontsize=8, ha='center', va='top',
                    rotation=90, fontweight='bold', alpha=0.85)
        for shift in regime_shifts:
            ax.axvline(x=shift['window'], color='black', linestyle='--', linewidth=0.9, alpha=0.7)

    title_suffix = ' (smoothed)' if smooth else ''
    ax.set_xlabel('Window')
    ax.set_ylabel('Expected Reward (mu_k^T * x_t)')
    ax.set_title('Expected Reward Trajectories Over Windows' + title_suffix)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc='upper left', ncol=2, frameon=False,
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout(pad=1.2)

    directory = f'myresults/Thomposon/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    fname = (f'expected_rewards_smoothed_{iterations}.png' if smooth
             else f'expected_rewards_{iterations}.png')
    plt.savefig(f'{directory}/{fname}', format='png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_selection_states(
    selection_states: List[str],
    dataset: str,
    entity: str,
    iterations: int,
) -> None:
    """
    Visualize the per-window selection state (random / exploitation / informed_exploration).

    Two stacked subplots:
      - Top: a per-window coloured timeline strip showing the state at each window.
      - Bottom: a bar chart with total counts and percentage labels.

    Saves to myresults/Thomposon/{dataset}/{entity}/selection_states_{iterations}.png.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    state_colours = {
        "random":               "#d62728",  # red
        "exploitation":         "#2ca02c",  # green
        "informed_exploration": "#1f77b4",  # blue
    }
    state_order = ["random", "exploitation", "informed_exploration"]

    T = len(selection_states)
    counts = {s: selection_states.count(s) for s in state_order}
    total = max(T, 1)

    fig, (ax_strip, ax_bar) = plt.subplots(
        2, 1, figsize=(10, 4), gridspec_kw={"height_ratios": [1, 3]}
    )

    # Top: timeline strip — one coloured cell per window
    strip_colours = [state_colours.get(s, "#888888") for s in selection_states]
    for t, c in enumerate(strip_colours):
        ax_strip.axvspan(t, t + 1, color=c, lw=0)
    ax_strip.set_xlim(0, max(T, 1))
    ax_strip.set_ylim(0, 1)
    ax_strip.set_yticks([])
    ax_strip.set_xlabel('Window')
    ax_strip.set_title('Selection State Timeline')

    # Bottom: bar chart of counts with percentage annotations
    bars = ax_bar.bar(
        state_order,
        [counts[s] for s in state_order],
        color=[state_colours[s] for s in state_order],
    )
    for bar, s in zip(bars, state_order):
        pct = 100.0 * counts[s] / total
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{counts[s]} ({pct:.1f}%)",
            ha='center', va='bottom', fontsize=10,
        )
    ax_bar.set_ylabel('Window Count')
    ax_bar.set_title('Selection State Totals')
    ax_bar.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # Legend outside the plot area
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=state_colours[s], label=s) for s in state_order
    ]
    fig.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )

    plt.tight_layout(pad=1.2)

    directory = f'myresults/Thomposon/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'{directory}/selection_states_{iterations}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()


def _top_k_models_by_norm(means: Dict[str, np.ndarray], k: int) -> List[str]:
    """Return the names of the top-k models by ||mu||^2."""
    ranked = sorted(
        {m: float(np.dot(mu.flatten(), mu.flatten())) for m, mu in means.items()}.items(),
        key=lambda x: x[1], reverse=True,
    )
    return [name for name, _ in ranked[:k]]


def _top_k_models_by_expected_reward(
    means: Dict[str, np.ndarray],
    contexts: List[np.ndarray],
    k: int,
) -> List[str]:
    """
    Return the top-k models by expected reward mu·x averaged over the given
    contexts. Uses the final mean vectors, so the ranking is consistent with the
    SHAP decomposition plotted alongside (whose per-channel values sum to
    mu·x - mu·baseline).
    """
    if not contexts or not means:
        return list(means.keys())[:k]
    totals: Dict[str, float] = {m: 0.0 for m in means}
    for ctx in contexts:
        for m, v in compute_expected_rewards(means, ctx).items():
            totals[m] += v
    ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:k]]


def _per_channel_shap_map(
    means: Dict[str, np.ndarray],
    top_models: List[str],
    context: np.ndarray,
    baseline: np.ndarray,
    n_channels: int,
) -> Dict[str, np.ndarray]:
    """Per-channel SHAP for each model at a single context (raw, signed)."""
    return {
        m: aggregate_shap_per_channel(
            compute_shap_values(means[m].flatten(), context, baseline), n_channels)
        for m in top_models
    }


def _avg_per_channel_shap_map(
    means: Dict[str, np.ndarray],
    top_models: List[str],
    contexts: List[np.ndarray],
    baseline: np.ndarray,
    n_channels: int,
    absolute: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Per-channel SHAP for each model, averaged over a list of contexts.

    absolute=False : raw signed average (meaningful for a regime, whose mean
                     context differs from the global baseline).
    absolute=True  : mean of |per-channel SHAP| — the standard SHAP global-
                     importance measure. Required for the whole-data average,
                     where the raw signed average is identically zero because
                     the baseline IS the mean of all contexts.
    """
    out: Dict[str, np.ndarray] = {}
    n = max(len(contexts), 1)
    for m in top_models:
        mu = means[m].flatten()
        acc = np.zeros(n_channels)
        for ctx in contexts:
            pc = aggregate_shap_per_channel(compute_shap_values(mu, ctx, baseline), n_channels)
            acc += np.abs(pc) if absolute else pc
        out[m] = acc / n
    return out


def _render_shap_comparison(
    per_channel_by_model: Dict[str, np.ndarray],
    top_models: List[str],
    top_n_channels: int,
    title: str,
    save_path: str,
) -> None:
    """
    Shared grouped-bar renderer for SHAP comparison plots. Channels shown are the
    union of each model's top_n_channels by |per-channel SHAP|. One bar per model
    per channel; legend placed outside the plot area.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    candidate_channels: set = set()
    for m in top_models:
        per_channel = per_channel_by_model[m]
        order = np.argsort(np.abs(per_channel))[::-1][:top_n_channels]
        candidate_channels.update(int(c) for c in order)
    channels = sorted(candidate_channels)
    if not channels:
        return

    n_models = len(top_models)
    bar_width = 0.8 / max(n_models, 1)
    x_base = np.arange(len(channels))
    colour_map = {name: plt.cm.tab20(i / max(n_models, 1)) for i, name in enumerate(top_models)}

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(channels) + 4), 5))
    for i, m in enumerate(top_models):
        vals = per_channel_by_model[m][channels]
        ax.bar(x_base + i * bar_width, vals, bar_width, label=m, color=colour_map[m])

    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks(x_base + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels([f"ch{c}" for c in channels], rotation=45, ha='right')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Per-channel SHAP contribution')
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(loc='upper left', frameon=False, bbox_to_anchor=(1.01, 1), borderaxespad=0)

    plt.tight_layout(pad=1.2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap_per_model(
    means: Dict[str, np.ndarray],
    shap_payload: Dict,
    dataset: str,
    entity: str,
    iterations: int,
    top_k_models: int = 3,
    top_n_channels: int = 10,
) -> None:
    """
    For the top_k_models (by ||mu||^2), draw horizontal bar charts of the
    top_n_channels with the largest |per-channel SHAP| contribution at the
    explanation context. Bars are coloured by sign (green > 0, red < 0).

    Saves to myresults/Thomposon/{dataset}/{entity}/shap_per_model_{iterations}.png.
    """
    if not shap_payload or shap_payload.get("n_channels", 0) <= 0:
        return

    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    context = shap_payload["explanation_context"]
    baseline = shap_payload["baseline_context"]
    n_channels = shap_payload["n_channels"]

    top_models = _top_k_models_by_norm(means, top_k_models)
    k = len(top_models)
    if k == 0:
        return

    fig, axes = plt.subplots(k, 1, figsize=(8, max(2.2 * k, 3.0)))
    if k == 1:
        axes = [axes]

    for ax, model_name in zip(axes, top_models):
        mu = means[model_name].flatten()
        shap_vals = compute_shap_values(mu, context, baseline)
        per_channel = aggregate_shap_per_channel(shap_vals, n_channels)
        e_r = float(np.dot(mu, context))

        order = np.argsort(np.abs(per_channel))[::-1][:top_n_channels]
        vals = per_channel[order]
        labels = [f"ch{c}" for c in order]
        colours = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]

        y_pos = np.arange(len(order))
        ax.barh(y_pos, vals, color=colours)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=0.6)
        ax.set_xlabel('Per-channel SHAP contribution')
        ax.set_title(f"{model_name}  |  E[R | last] = {e_r:+.4f}")
        ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.6)

    plt.tight_layout(pad=1.2)
    directory = f'myresults/Thomposon/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'{directory}/shap_per_model_{iterations}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap_comparison(
    means: Dict[str, np.ndarray],
    shap_payload: Dict,
    dataset: str,
    entity: str,
    iterations: int,
    top_k_models: int = 3,
    top_n_channels: int = 8,
    all_models: bool = False,
) -> None:
    """
    Grouped bar chart comparing models on the channels most relevant to their
    disagreement at the last window. Channels shown are the union of each model's
    top_n_channels by |per-channel SHAP|.

    all_models : bool
        When False (default) the top_k_models by ||mu||^2 are shown, saved as
        shap_comparison_{iterations}.png. When True every model is shown, saved
        as shap_comparison_all_{iterations}.png.
    """
    if not shap_payload or shap_payload.get("n_channels", 0) <= 0:
        return

    context = shap_payload["explanation_context"]
    baseline = shap_payload["baseline_context"]
    n_channels = shap_payload["n_channels"]

    if all_models:
        sel_models = _top_k_models_by_norm(means, len(means))
        suffix, title = '_all', 'SHAP Comparison Across All Models (at last window)'
    else:
        sel_models = _top_k_models_by_norm(means, top_k_models)
        suffix, title = '', 'SHAP Comparison Across Top Models (at last window)'
    if not sel_models:
        return

    per_channel = _per_channel_shap_map(means, sel_models, context, baseline, n_channels)
    _render_shap_comparison(
        per_channel, sel_models, top_n_channels,
        title=title,
        save_path=f'myresults/Thomposon/{dataset}/{entity}/shap_comparison{suffix}_{iterations}.png',
    )


def plot_shap_per_window(
    means: Dict[str, np.ndarray],
    shap_payload: Dict,
    dataset: str,
    entity: str,
    iterations: int,
    top_k_models: int = 3,
    top_n_channels: int = 8,
    all_models: bool = False,
    sample_stride: Optional[int] = None,
) -> None:
    """
    One SHAP comparison plot per window (raw signed per-channel SHAP), in the
    same style as shap_comparison.

    all_models : bool
        When False (default) each window's plot shows the top_k_models by
        expected reward (mu·x) at that window's context; saved under
        shap_per_window_{iterations}/. When True every model is shown; saved
        under shap_per_window_all_{iterations}/.
    sample_stride : Optional[int]
        When set (e.g. 10) only every `sample_stride`-th window is plotted
        (window 0, 10, 20, …), saved under shap_per_window_every{stride}_{iterations}/.
        Useful for a quick overview without one plot per window.
    """
    if not shap_payload or shap_payload.get("n_channels", 0) <= 0:
        return
    contexts = shap_payload.get("all_contexts", [])
    if not contexts or not means:
        return
    baseline = shap_payload["baseline_context"]
    n_channels = shap_payload["n_channels"]

    every_model = _top_k_models_by_norm(means, len(means)) if all_models else None
    base = "shap_per_window_all" if all_models else "shap_per_window"
    if sample_stride and sample_stride > 1:
        base = f"{base}_every{sample_stride}"
    directory = f'myresults/Thomposon/{dataset}/{entity}/{base}_{iterations}/'
    os.makedirs(directory, exist_ok=True)
    for t, ctx in enumerate(contexts):
        if sample_stride and sample_stride > 1 and t % sample_stride != 0:
            continue
        if all_models:
            sel_models = every_model
            title = f'SHAP — window {t} (all models)'
        else:
            sel_models = _top_k_models_by_expected_reward(means, [ctx], top_k_models)
            title = f'SHAP — window {t} (top {top_k_models} by E[R] in window)'
        per_channel = _per_channel_shap_map(means, sel_models, ctx, baseline, n_channels)
        _render_shap_comparison(
            per_channel, sel_models, top_n_channels,
            title=title,
            save_path=os.path.join(directory, f'window_{t:03d}.png'),
        )


def plot_shap_per_regime(
    means: Dict[str, np.ndarray],
    shap_payload: Dict,
    regime_shifts: List[Dict],
    dataset: str,
    entity: str,
    iterations: int,
    top_k_models: int = 3,
    top_n_channels: int = 8,
    all_models: bool = False,
) -> None:
    """
    One SHAP comparison plot per regime, showing the raw signed per-channel SHAP
    averaged over that regime's windows. A regime's mean context differs from the
    global baseline, so the signed average is meaningful (and direction-bearing).

    all_models : bool
        When False (default) each regime's plot shows the top_k_models by
        expected reward (mu·x) averaged over that regime's windows; saved under
        shap_per_regime_{iterations}/. When True every model is shown; saved
        under shap_per_regime_all_{iterations}/.
    """
    if not shap_payload or shap_payload.get("n_channels", 0) <= 0:
        return
    contexts = shap_payload.get("all_contexts", [])
    if not contexts or not means:
        return
    baseline = shap_payload["baseline_context"]
    n_channels = shap_payload["n_channels"]

    fallback = _top_k_models_by_norm(means, 1)[0]
    segments = reconstruct_regime_segments(regime_shifts, len(contexts),
                                           fallback_model=fallback)
    every_model = _top_k_models_by_norm(means, len(means)) if all_models else None
    folder = f'shap_per_regime_all_{iterations}' if all_models else f'shap_per_regime_{iterations}'
    directory = f'myresults/Thomposon/{dataset}/{entity}/{folder}/'
    os.makedirs(directory, exist_ok=True)
    for i, (start, end, model, _duration) in enumerate(segments, 1):
        regime_ctx = contexts[start:end + 1]
        if not regime_ctx:
            continue
        if all_models:
            sel_models = every_model
            title = f'SHAP — regime {i} ({model}, windows {start}-{end}, all models)'
        else:
            sel_models = _top_k_models_by_expected_reward(means, regime_ctx, top_k_models)
            title = (f'SHAP — regime {i} ({model}, windows {start}-{end}, '
                     f'top {top_k_models} by E[R] in regime)')
        per_channel = _avg_per_channel_shap_map(
            means, sel_models, regime_ctx, baseline, n_channels, absolute=False)
        _render_shap_comparison(
            per_channel, sel_models, top_n_channels,
            title=title,
            save_path=os.path.join(directory, f'regime_{i:02d}_w{start}-{end}_{model}.png'),
        )


def plot_shap_average_all(
    means: Dict[str, np.ndarray],
    shap_payload: Dict,
    dataset: str,
    entity: str,
    iterations: int,
    top_k_models: int = 3,
    top_n_channels: int = 8,
    all_models: bool = True,
) -> None:
    """
    A single SHAP comparison plot summarising the whole run: the mean of
    |per-channel SHAP| across all windows (the standard SHAP global-importance
    measure). The raw signed average over all windows is identically zero — the
    baseline IS the mean of all contexts — so absolute values are used here.

    all_models : bool
        When True (default) every model is shown, saved as
        shap_average_all_{iterations}.png. When False only the top_k_models by
        expected reward (mu·x) averaged over the whole run are shown, saved as
        shap_average_top3_{iterations}.png.
    """
    if not shap_payload or shap_payload.get("n_channels", 0) <= 0:
        return
    contexts = shap_payload.get("all_contexts", [])
    if not contexts or not means:
        return
    baseline = shap_payload["baseline_context"]
    n_channels = shap_payload["n_channels"]

    if all_models:
        # Every model, ordered by ||mu||^2 for a stable, meaningful legend order.
        sel_models = _top_k_models_by_norm(means, len(means))
        suffix = 'all'
        title = 'Mean |SHAP| Across All Windows — all models (global importance)'
    else:
        sel_models = _top_k_models_by_expected_reward(means, contexts, top_k_models)
        suffix = f'top{top_k_models}'
        title = (f'Mean |SHAP| Across All Windows — top {top_k_models} by E[R] '
                 '(global importance)')

    per_channel = _avg_per_channel_shap_map(
        means, sel_models, contexts, baseline, n_channels, absolute=True)
    _render_shap_comparison(
        per_channel, sel_models, top_n_channels,
        title=title,
        save_path=f'myresults/Thomposon/{dataset}/{entity}/shap_average_{suffix}_{iterations}.png',
    )


def explain_thompson_sampling(
    means: Dict[str, np.ndarray],
    expected_rewards_history: Dict[str, List[float]],
    l2_norm_history: Dict[str, List[float]],
    pre_expected_rewards_history: Dict[str, List[float]],
    list_of_chosen_models: List[str],
    regime_shifts: List[Dict],
    blip_windows: List[str],
    selection_states: List[str],
    shap_payload: Optional[Dict],
    dataset: str,
    entity: str,
    iterations: int,
) -> None:
    """
    Write a structured plain-text explainability report to disk.

    Sections: header, per-window table (chosen model, dominant model, top expected
    reward, selection state), regime summary, shift events, blips, selection state
    summary, SHAP feature attribution (when shap_payload is provided), SHAP preference
    decomposition, and final ranking by ||mu_k||^2.

    The Dominant / Top E[Reward] columns are computed from pre_expected_rewards_history
    — the expected rewards at the *time of sampling* (pre-update means). When every
    model's expected reward is identical (e.g. window 0, all means still the zero
    vector) there is no meaningful winner, so both columns print 'N/A'.

    Saves to myresults/Thomposon/{dataset}/{entity}/explainability_{iterations}.txt.
    """
    model_list = list(expected_rewards_history.keys())
    T = len(list_of_chosen_models)

    # Per-window dominant model + top expected reward, computed from the PRE-update
    # means (the beliefs at the time of sampling). 'N/A' when all rewards are tied
    # (e.g. window 0, where every mean vector is still the zero vector).
    dominant_per_window: List[str] = []
    top_reward_per_window: List[Optional[float]] = []
    for t in range(T):
        rewards_at_t = {
            m: pre_expected_rewards_history[m][t]
            for m in model_list
            if t < len(pre_expected_rewards_history[m]) and not np.isnan(pre_expected_rewards_history[m][t])
        }
        if rewards_at_t and max(rewards_at_t.values()) != min(rewards_at_t.values()):
            dom = max(rewards_at_t, key=rewards_at_t.get)
            dominant_per_window.append(dom)
            top_reward_per_window.append(rewards_at_t[dom])
        else:
            dominant_per_window.append('N/A')
            top_reward_per_window.append(None)

    # Reconstruct regime segments from shift events. The no-shift fallback uses the
    # most frequent dominant model (window 0 is 'N/A' and not a meaningful fallback).
    _valid_doms = [d for d in dominant_per_window if d != 'N/A']
    first_dom = max(set(_valid_doms), key=_valid_doms.count) if _valid_doms else 'N/A'
    regime_segments = reconstruct_regime_segments(regime_shifts, T, fallback_model=first_dom)

    # Per-regime story blocks: regime-mean expected rewards (from the recorded
    # pre-update beliefs), the leader's SHAP channels on the regime-aggregated
    # context, and the leader-vs-runner-up preference decomposition. Computed
    # ONCE here and consumed by BOTH the report below and the Intermediate
    # Representation, so the two always match.
    regimes_data: List[Dict[str, Any]] = []
    try:
        all_ctx = (shap_payload or {}).get("all_contexts") or []
        base_ctx = (shap_payload or {}).get("baseline_context")
        n_ch = int((shap_payload or {}).get("n_channels", 0) or 0)
        for seg_idx, (seg_s, seg_e, seg_m, seg_dur) in enumerate(regime_segments):
            seg_end = min(int(seg_e), T - 1)
            seg_start = max(int(seg_s), 0)
            reg_rewards: Dict[str, float] = {}
            for m in model_list:
                hist = pre_expected_rewards_history.get(m, [])
                vals = [hist[t] for t in range(seg_start, seg_end + 1)
                        if t < len(hist) and hist[t] is not None and not np.isnan(hist[t])]
                if vals:
                    reg_rewards[m] = float(np.mean(vals))
            top3 = sorted(reg_rewards.items(), key=lambda kv: kv[1], reverse=True)[:3]
            gap = (top3[0][1] - top3[1][1]) if len(top3) >= 2 else float('nan')
            leader = seg_m if seg_m in means else (top3[0][0] if top3 else None)
            runner = next((m for m, _ in top3 if m != leader), None)

            # Channel contributions are split by SIGN before truncation: a
            # magnitude-sorted signed list invites "driven by" phrasings whose
            # top entries actually push the other way. Top-3 per direction.
            def _split_by_sign(vals: np.ndarray) -> Tuple[list, list]:
                pos = [(int(c), float(vals[c])) for c in np.argsort(vals)[::-1]
                       if vals[c] > 0][:3]
                neg = [(int(c), float(vals[c])) for c in np.argsort(vals)
                       if vals[c] < 0][:3]
                return pos, neg

            shap_raising = shap_lowering = None
            pref_favor_leader = pref_favor_runner = None
            pref_gap = float('nan')
            if leader is not None and n_ch > 0 and len(all_ctx) > seg_end:
                ctx_reg = np.mean([np.asarray(all_ctx[t], dtype=float)
                                   for t in range(seg_start, seg_end + 1)], axis=0)
                mu_l = means[leader].flatten()
                pc_l = aggregate_shap_per_channel(
                    compute_shap_values(mu_l, ctx_reg, base_ctx), n_ch)
                shap_raising, shap_lowering = _split_by_sign(pc_l)
                if runner is not None and runner in means:
                    mu_r = means[runner].flatten()
                    pc_r = aggregate_shap_per_channel(
                        compute_shap_values(mu_r, ctx_reg, base_ctx), n_ch)
                    delta = pc_l - pc_r
                    pref_gap = float(np.dot(mu_l - mu_r, ctx_reg))
                    pref_favor_leader, pref_favor_runner = _split_by_sign(delta)

            regimes_data.append({
                "index": seg_idx, "start": seg_start, "end": seg_end,
                "duration": int(seg_dur), "leader": leader,
                "rewards_top": top3, "reward_gap": gap, "runner_up": runner,
                "shap_raising": shap_raising, "shap_lowering": shap_lowering,
                "pref_favor_leader": pref_favor_leader,
                "pref_favor_runner": pref_favor_runner,
                "pref_gap": pref_gap,
            })
    except Exception as e:
        logger.error(f"Thompson per-regime computation failed (non-fatal): {e}")
        regimes_data = []

    directory = f'myresults/Thomposon/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(directory, f'explainability_{iterations}.txt')

    with open(output_file, 'w') as f:
        f.write("=== Thompson Sampling Explainability Report ===\n")
        f.write(f"Dataset: {dataset}  |  Entity: {entity}  |  Windows: {T}\n\n")

        f.write("--- Per-Window Summary ---\n")
        f.write("(Dominant / Top E[Reward] reflect the beliefs at the time of sampling)\n")
        f.write(f"{'Window':>8}  {'Chosen':>12}  {'Dominant':>12}  {'Top E[Reward]':>14}  {'State':>22}\n")
        f.write("-" * 76 + "\n")
        for t in range(T):
            chosen = list_of_chosen_models[t] if t < len(list_of_chosen_models) else 'N/A'
            dominant = dominant_per_window[t]
            top_reward = top_reward_per_window[t]
            top_str = f"{top_reward:.4f}" if top_reward is not None else "N/A"
            state = selection_states[t] if t < len(selection_states) else 'N/A'
            f.write(f"{t:>8}  {chosen:>12}  {dominant:>12}  {top_str:>14}  {state:>22}\n")

        f.write("\n--- Regime Summary ---\n")
        f.write(f"{'Start':>8}  {'End':>8}  {'Model':>12}  {'Duration':>10}\n")
        f.write("-" * 44 + "\n")
        for rs, re, rm, dur in regime_segments:
            f.write(f"{rs:>8}  {re:>8}  {rm:>12}  {dur:>10} windows\n")

        f.write("\n--- Regime Shift Events ---\n")
        if regime_shifts:
            f.write(f"{'Window':>8}  {'From':>12}  {'To':>12}  {'Delta':>10}  {'Old Regime Len':>16}\n")
            f.write("-" * 64 + "\n")
            for s in regime_shifts:
                f.write(f"{s['window']:>8}  {s['from_model']:>12}  {s['to_model']:>12}  "
                        f"{s['reward_delta']:>10.4f}  {s['regime_length']:>16} windows\n")
        else:
            f.write("No regime shifts detected.\n")

        f.write("\n--- Brief Blips ---\n")
        if blip_windows:
            for b in blip_windows:
                f.write(f"  {b}\n")
        else:
            f.write("No blips detected.\n")

        f.write("\n--- Per-Regime Expected Rewards & SHAP ---\n")
        f.write("(Mean E[reward] over each regime's windows from the recorded pre-update\n")
        f.write(" beliefs; SHAP attributes the FINAL posterior means on the regime's\n")
        f.write(" aggregated context. Matches the Intermediate Representation.)\n")
        if not regimes_data:
            f.write("Not available.\n")
        for r in regimes_data:
            f.write(f"\nRegime {r['index']}: windows {r['start']}-{r['end']} "
                    f"({r['duration']} windows), led by {r['leader']}\n")
            if r["rewards_top"]:
                rw = ", ".join(f"{m} {v:+.4f}" for m, v in r["rewards_top"])
                gap_s = (f";  leader-vs-runner-up mean-reward gap {r['reward_gap']:+.4f}"
                         if not np.isnan(r["reward_gap"]) else "")
                f.write(f"  Mean E[reward]: {rw}{gap_s}\n")
            if r["shap_raising"] or r["shap_lowering"]:
                raise_s = ", ".join(f"ch {c} {v:+.4f}"
                                    for c, v in (r["shap_raising"] or [])) or "none"
                lower_s = ", ".join(f"ch {c} {v:+.4f}"
                                    for c, v in (r["shap_lowering"] or [])) or "none"
                f.write(f"  Channels raising {r['leader']}'s E[reward]: {raise_s}\n")
                f.write(f"  Channels lowering {r['leader']}'s E[reward]: {lower_s}\n")
            has_pref = r["pref_favor_leader"] or r["pref_favor_runner"]
            if has_pref and r["runner_up"] and not np.isnan(r["pref_gap"]):
                favored = r["leader"] if r["pref_gap"] >= 0 else r["runner_up"]
                fl = ", ".join(f"ch {c} {d:+.4f}"
                               for c, d in (r["pref_favor_leader"] or [])) or "none"
                fr = ", ".join(f"ch {c} {d:+.4f}"
                               for c, d in (r["pref_favor_runner"] or [])) or "none"
                f.write(f"  Preference {r['leader']} vs {r['runner_up']}: linear "
                        f"preference score at the regime-average context favors "
                        f"{favored} by {abs(r['pref_gap']):.4f}\n")
                f.write(f"    channels favoring {r['leader']}: {fl}\n")
                f.write(f"    channels favoring {r['runner_up']}: {fr}\n")

        f.write("\n--- Selection State Summary ---\n")
        state_order = ["random", "exploitation", "informed_exploration"]
        state_counts = {s: selection_states.count(s) for s in state_order}
        state_total = max(len(selection_states), 1)
        for s in state_order:
            pct = 100.0 * state_counts[s] / state_total
            f.write(f"  {s:<22}: {state_counts[s]:>5} ({pct:5.1f}%)\n")

        # SHAP feature attribution & cross-model preference decomposition
        if shap_payload and shap_payload.get("n_channels", 0) > 0:
            ctx = shap_payload["explanation_context"]
            base = shap_payload["baseline_context"]
            n_ch = shap_payload["n_channels"]

            f.write("\n--- SHAP Feature Attribution (final mean vectors) ---\n")
            f.write("Explanation context : last window (L2-normalised)\n")
            f.write("Baseline            : mean over all L2-normalised windows\n")
            f.write("Per-feature phi_i   = mu_i * (x_i - baseline_i)\n")
            f.write("Per-channel         = sum of phi_i over the channel's window timesteps\n\n")

            top_models = _top_k_models_by_norm(means, 3)
            per_channel_by_model: Dict[str, np.ndarray] = {}
            for rank, model_name in enumerate(top_models, 1):
                mu = means[model_name].flatten()
                shap_vals = compute_shap_values(mu, ctx, base)
                per_ch = aggregate_shap_per_channel(shap_vals, n_ch)
                per_channel_by_model[model_name] = per_ch
                e_r = float(np.dot(mu, ctx))
                e_r_base = float(np.dot(mu, base))
                delta = e_r - e_r_base
                f.write(f"  {rank}. {model_name}  "
                        f"(E[R | last] = {e_r:+.4f},  baseline E[R] = {e_r_base:+.4f},  delta = {delta:+.4f})\n")
                f.write(f"     Top 5 channels by |per-channel SHAP|:\n")
                top_idx = np.argsort(np.abs(per_ch))[::-1][:5]
                for c in top_idx:
                    f.write(f"       channel {int(c):>3} : {per_ch[c]:+.4f}\n")
                f.write(f"     Sum over all channels: {float(per_ch.sum()):+.4f}\n\n")

            if len(top_models) >= 2:
                top, second = top_models[0], top_models[1]
                gap = float(np.dot(means[top].flatten() - means[second].flatten(), ctx))
                delta_per_ch = per_channel_by_model[top] - per_channel_by_model[second]
                f.write("--- SHAP Preference Decomposition ---\n")
                f.write(f"Top model: {top}  vs  2nd: {second}\n")
                f.write(f"Preference gap at last window: (mu_{top} - mu_{second})^T x_last = {gap:+.4f}\n")
                f.write("Top 5 channels driving the preference:\n")
                top_idx = np.argsort(np.abs(delta_per_ch))[::-1][:5]
                for c in top_idx:
                    a = per_channel_by_model[top][c]
                    b = per_channel_by_model[second][c]
                    f.write(f"  channel {int(c):>3} : "
                            f"{top}={a:+.4f}  {second}={b:+.4f}  delta={a - b:+.4f}\n")

        f.write("\n--- Final Model Ranking (by ||mu_k||^2) ---\n")
        f.write(f"  {'Rank':>4}  {'Model':>12}  {'Final Score':>12}  {'Peak Score':>12}\n")
        f.write("  " + "-" * 46 + "\n")
        final_scores = {m: float(np.dot(mu.flatten(), mu.flatten())) for m, mu in means.items()}
        peak_scores = {
            m: float(np.nanmax(l2_norm_history[m])) if l2_norm_history.get(m) else 0.0
            for m in means
        }
        ranking = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (m, score) in enumerate(ranking, 1):
            f.write(f"  {rank:>4}  {m:>12}  {score:>12.6f}  {peak_scores.get(m, 0.0):>12.6f}\n")

    print(f"Explainability report saved to {output_file}")

    # ── Intermediate Representation (grounded LLM input; non-fatal) ─────────
    # Reuses regimes_data computed above — the report and the IR always match.
    try:
        _ir = _ir_module()
        ir_doc = _ir.build_thompson_ir(
            dataset, entity, n_windows=T,
            final_ranking=ranking,
            regimes=regimes_data,
            shifts=regime_shifts,
            blip_count=len(blip_windows),
            state_fractions={s: state_counts[s] / state_total for s in state_counts},
            final_state=selection_states[-1] if selection_states else "not_available",
        )
        _ir.write_stage_ir(ir_doc, dataset, entity, "ir_thompson")
    except Exception as e:
        logger.error(f"Thompson IR emission failed (non-fatal): {e}")


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
                                 initial_epsilon=0.2, epsilon_decay=0.99, f1_weight=0.5, pr_auc_weight=0.5,
                                 explain=False):
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
    _fit_result = fit_linear_thompson_sampling(
        test_data,
        trained_models,
        test_data.entities[0].Y,
        test_data.entities[0].labels,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        f1_weight=f1_weight,
        pr_auc_weight=pr_auc_weight,
        iterations=iterations,
        explain=explain,
    )
    if explain:
        (means, covariances, history, list_of_chosen_models,
         exp_rewards_hist, l2_norm_hist, selection_states,
         pre_exp_rewards_hist, shap_payload) = _fit_result
    else:
        means, covariances, history, list_of_chosen_models = _fit_result

    # Rank models
    ranked_models = rank_models(means)

    directory = f'myresults/Thomposon/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(directory, f"thompson_sampling_results_{dataset}_{entity}_{iterations}_{iteration}.txt")

    # Plot history
    plot_history(history, trained_models, dataset, entity, iterations)

    if explain:
        regime_shifts, blip_windows = detect_regime_shifts(exp_rewards_hist)
        plot_expected_rewards(exp_rewards_hist, regime_shifts, list(trained_models.keys()),
                              dataset, entity, iterations, smooth=False)
        plot_expected_rewards(exp_rewards_hist, regime_shifts, list(trained_models.keys()),
                              dataset, entity, iterations, smooth=True)
        plot_selection_states(selection_states, dataset, entity, iterations)
        plot_shap_per_model(means, shap_payload, dataset, entity, iterations)
        # Each SHAP comparison plot is produced in both a top-k and an all-models variant.
        plot_shap_comparison(means, shap_payload, dataset, entity, iterations)
        plot_shap_comparison(means, shap_payload, dataset, entity, iterations, all_models=True)
        plot_shap_per_window(means, shap_payload, dataset, entity, iterations)
        plot_shap_per_window(means, shap_payload, dataset, entity, iterations, all_models=True)
        plot_shap_per_window(means, shap_payload, dataset, entity, iterations, sample_stride=10)
        plot_shap_per_regime(means, shap_payload, regime_shifts, dataset, entity, iterations)
        plot_shap_per_regime(means, shap_payload, regime_shifts, dataset, entity, iterations,
                             all_models=True)
        plot_shap_average_all(means, shap_payload, dataset, entity, iterations)
        plot_shap_average_all(means, shap_payload, dataset, entity, iterations, all_models=False)
        explain_thompson_sampling(means, exp_rewards_hist, l2_norm_hist, pre_exp_rewards_hist,
                                  list_of_chosen_models,
                                  regime_shifts, blip_windows, selection_states,
                                  shap_payload,
                                  dataset, entity, iterations)

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
            f.write(f"  Mean: {np.array2string(mean, threshold=np.inf, max_line_width=np.inf)}\n")
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
