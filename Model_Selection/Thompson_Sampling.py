import copy
import json
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

from Utils.pipeline_spec import abbreviate_detector
from Utils.plot_labels import draw_abbreviation_key

from Metrics.Ensemble_GA import evaluate_individual_models
from Metrics.Ensemble_GA import evaluate_model_consistently
from Utils.model_selection_utils import timed_out_detectors
from Utils.pipeline_spec import (DEFAULT_DECISION_METRICS, combine_metrics,
                                 metrics_required)
from Metrics.metrics import vus_score, vus_window
from Metrics.metrics import prauc, range_based_precision_recall_f1_auc
from Explainability import ir


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


def aggregate_shap_per_context_feature(shap_values: np.ndarray, n_context_features: int) -> np.ndarray:
    """
    Sum SHAP values within each context feature's window-of-timesteps slice.

    The flattened feature vector is assumed to be reshape(n_context_features, window_size)
    flattened in C-order (numpy default), so feature i belongs to context feature i // window_size.

    If shap_values.size is not divisible by n_context_features (should not happen after the
    n_features fix), trailing entries are dropped.

    Parameters
    ----------
    shap_values : np.ndarray
        Per-feature SHAP values, shape (d,).
    n_context_features : int
        Number of context features.

    Returns
    -------
    np.ndarray
        Per-context-feature contributions, shape (n_context_features,).
    """
    if n_context_features <= 0:
        return np.zeros(0)
    window_size = shap_values.size // n_context_features
    if window_size == 0:
        return np.zeros(n_context_features)
    return shap_values[: n_context_features * window_size].reshape(n_context_features, window_size).sum(axis=1)


def reward_contribution_per_context_feature(mean: np.ndarray, context: np.ndarray,
                                    n_context_features: int) -> np.ndarray:
    """
    Split the expected reward mu^T x into one contribution per context feature.

    contrib(c) = sum over context feature c's timesteps of mu_i * x_i, so the parts sum
    to mu^T x EXACTLY — the model has no intercept, so there is no remainder.

    This is the honest answer to "how much does this context feature contribute to this
    detector's expected reward". SHAP answers a different question: it measures
    each context feature's deviation from a TYPICAL window, so it explains only
    mu^T x - mu^T baseline and discards the constant mu^T baseline, which is
    usually the bulk of the prediction. Worse for any averaged view, the signed
    SHAP average over all windows is identically zero by construction, because
    the baseline IS the mean of those windows.

    Signed: mu and the (L2-normalised) context can both be negative, so a
    context feature can pull the expected reward down.
    """
    if n_context_features <= 0:
        return np.zeros(0)
    mu, x = mean.flatten(), context.flatten()
    d = min(mu.size, x.size)
    window_size = d // n_context_features
    if window_size == 0:
        return np.zeros(n_context_features)
    keep = n_context_features * window_size
    return (mu[:keep] * x[:keep]).reshape(n_context_features, window_size).sum(axis=1)


def _per_context_feature_reward_map(
    means: Dict[str, np.ndarray],
    top_models: List[str],
    context: np.ndarray,
    n_context_features: int,
) -> Dict[str, np.ndarray]:
    """Per-context-feature expected-reward contribution for each model at one context."""
    return {m: reward_contribution_per_context_feature(means[m], context, n_context_features)
            for m in top_models}


def _avg_per_context_feature_reward_map(
    means: Dict[str, np.ndarray],
    top_models: List[str],
    contexts: List[np.ndarray],
    n_context_features: int,
    means_per_context: Optional[List[Dict[str, np.ndarray]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Per-context-feature reward contribution averaged over a list of contexts.

    No `absolute` switch, unlike the SHAP version: this average is not
    structurally zero, so the signed mean is the meaningful quantity and there
    is nothing to work around. Averaged over every window it is each context feature's
    share of the detector's expected reward on a typical window.

    `means_per_context` explains window t with the beliefs held at window t;
    the product mu*x is bilinear, so the per-window loop is the correct order.
    """
    out: Dict[str, np.ndarray] = {}
    n = max(len(contexts), 1)
    for m in top_models:
        mu_fixed = means[m].flatten() if means_per_context is None else None
        acc = np.zeros(n_context_features)
        for i, ctx in enumerate(contexts):
            if mu_fixed is not None:
                mu = mu_fixed
            else:
                if i >= len(means_per_context) or m not in means_per_context[i]:
                    continue
                mu = np.asarray(means_per_context[i][m]).flatten()
            acc += reward_contribution_per_context_feature(mu, ctx, n_context_features)
        out[m] = acc / n
    return out


def aggregate_squared_per_context_feature(mean: np.ndarray, n_context_features: int) -> np.ndarray:
    """
    Split the ranking score ||mu||^2 into one contribution per context feature.

    The ranking criterion is mu^T mu = sum_i mu_i^2, and the flattened feature
    vector lays context features out in contiguous blocks of window_size timesteps (the
    same layout aggregate_shap_per_context_feature assumes), so context feature c contributes

        contrib(c) = sum_{i in block c} mu_i^2

    and sum_c contrib(c) == ||mu||^2 exactly. Unlike a SHAP attribution this is
    a sum of squares, so **every contribution is non-negative**: it says how a
    detector's score is divided among context features, never which context features pushed it
    down. Only a comparison between two detectors (rank_gap_decomposition) is
    signed.

    Parameters
    ----------
    mean : np.ndarray
        The posterior mean mu_k. May be 1-D (d,) or column (d, 1).
    n_context_features : int
        Number of context features.

    Returns
    -------
    np.ndarray
        Per-context-feature contributions to ||mu||^2, shape (n_context_features,), all >= 0.
    """
    if n_context_features <= 0:
        return np.zeros(0)
    mu = mean.flatten()
    window_size = mu.size // n_context_features
    if window_size == 0:
        return np.zeros(n_context_features)
    block = mu[: n_context_features * window_size].reshape(n_context_features, window_size)
    return np.square(block).sum(axis=1)


def rank_gap_decomposition(mean_a: np.ndarray, mean_b: np.ndarray,
                           n_context_features: int) -> np.ndarray:
    """
    Split the ranking gap ||mu_a||^2 - ||mu_b||^2 into one term per context feature.

    This is the signed counterpart of aggregate_squared_per_context_feature: it sums to
    the margin exactly, and a negative entry means that context feature was worked in
    B's favour, i.e. it cost A part of its lead. This — not the per-detector
    split — is what answers "which context features put A ahead of B".

    Returns
    -------
    np.ndarray
        Per-context-feature signed contributions to the gap, shape (n_context_features,).
    """
    return (aggregate_squared_per_context_feature(mean_a, n_context_features)
            - aggregate_squared_per_context_feature(mean_b, n_context_features))


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


def leadership_regimes(
    means_history: List[Dict[str, np.ndarray]],
    warmup: int = 10,
) -> Tuple[List[Tuple[int, int, str, int]], int]:
    """
    Segment the run by whichever detector leads on the ranking score ||mu||^2.

    Deliberately simpler than detect_regime_shifts(), which smooths expected
    rewards and imposes a minimum length: here a regime is just a maximal run of
    consecutive windows with the same argmax_k ||mu_k||^2 — no smoothing, no
    minimum length. ||mu||^2 leadership is far stickier than expected-reward
    leadership (it only moves in the windows where that arm was selected), so
    plain run-length encoding already yields a handful of regimes rather than
    one per handful of windows.

    The first `warmup` windows are excluded: until every arm has been sampled at
    least once most scores are still exactly zero, so leadership there is an
    artefact of sampling order rather than a finding. The warm-up collapses to 0
    on runs too short to spare it, and the value actually used is returned so
    the explanation can state it instead of asserting a constant.

    Reads the PRE-update means, keeping this stage on the single vintage the fit
    loop snapshots (see _pre_means_hist) — every quantity here describes the
    same moment. The post-update norms in _l2_norm_hist are the alternative.

    Ties are resolved in favour of the incumbent leader, so an exact tie (all
    scores zero, in practice) never manufactures a boundary; with no incumbent
    the first name in sorted order wins, which keeps the result deterministic.

    Parameters
    ----------
    means_history : List[Dict[str, np.ndarray]]
        Per-window posterior means, one dict of {model: mu} per window.
    warmup : int
        Number of leading windows to exclude.

    Returns
    -------
    Tuple[List[Tuple[int, int, str, int]], int]
        (segments, warmup_used), segments as (start, end_inclusive, leader,
        duration) with window indices on the original, un-trimmed timeline.
    """
    T = len(means_history)
    if T <= 0:
        return [], 0
    warmup_used = warmup if (warmup > 0 and T > warmup + 2) else 0

    segments: List[Tuple[int, int, str, int]] = []
    leader: Optional[str] = None
    start = warmup_used
    for t in range(warmup_used, T):
        scores = {m: float(np.dot(mu.flatten(), mu.flatten()))
                  for m, mu in means_history[t].items()}
        finite = {m: s for m, s in scores.items() if np.isfinite(s)}
        if not finite:
            # Every score unusable; carry the incumbent rather than inventing a
            # boundary out of a failed window.
            if leader is None:
                continue
            current = leader
        else:
            best = max(finite.values())
            if leader is not None and np.isclose(finite.get(leader, -np.inf), best):
                current = leader
            else:
                current = min(m for m, s in finite.items() if np.isclose(s, best))

        if leader is None:
            leader, start = current, t
        elif current != leader:
            segments.append((start, t - 1, leader, t - start))
            leader, start = current, t

    if leader is not None:
        segments.append((start, T - 1, leader, T - start))
    return segments, warmup_used


def fit_linear_thompson_sampling(dataset,
                                 models: Dict[str, Any], data: np.ndarray, targets: np.ndarray,
                                 initial_epsilon: float = 0.2,
                                 epsilon_decay: float = 0.99, f1_weight: float = 0.7, pr_auc_weight: float = 0.3,
                                 iterations: int = 100,
                                 explain: bool = False, metrics=DEFAULT_DECISION_METRICS,
                                 vus_win=None) -> Tuple[
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

    # n_features should be the flattened window length (n_context_features * window_size)
    # data_windows elements have shape (n_context_features, window_size), so flatten to get full feature length
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
    # Posterior means as they stood BEFORE each window's update — the beliefs that
    # informed the decision. Every explainability quantity (regime detection,
    # selection states, SHAP) reads from this one vintage so the stage never
    # compares a detector against itself at two different points in its learning.
    _pre_means_hist: List[Dict[str, np.ndarray]] = []

    for iteration in range(num_windows):
        logger.info(f"Iteration {iteration + 1}")
        # Offered to the sampler, but never removed from `models`: every
        # per-model history keeps one entry per window, including the nan rows
        # the error path below writes.
        candidates = {m: v for m, v in models.items()
                      if m not in timed_out_detectors()}
        if not candidates:
            logger.error("Every detector has been killed for slowness; "
                         "no arm left to sample.")
            break
        try:
            # Pass the current window as context so selection uses theta_tilde^T * x.
            # Normalise to unit length so that datasets with large sensor values
            # (e.g. SMD with 38 context features) do not cause xxᵀ to explode and collapse Σ.
            context = data_windows[iteration].flatten()
            context = context / (np.linalg.norm(context) + 1e-10)
            chosen_model_name, was_random = sample_model(candidates, means, covariances, epsilon, context)
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
            # Snapshot (copy — means are mutated in place) alongside the rewards
            # so the two stay index-aligned even when a window is skipped.
            _pre_means_hist.append({_m: means[_m].flatten().copy() for _m in models})
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

            # MUST be the range-based metric, not `f1_score`. `f1_score` counts
            # TP as `sum(predict * actual)`, so it is an F1 only when `predict`
            # is 0/1; handed raw scores it still computes, but it is unbounded
            # once a score can go negative — which LSTMVAE's Gaussian NLL does
            # whenever sigma < 1. The range-based metric thresholds internally
            # over its own sweep, so it never mistakes a score for a prediction.
            # Costs one sweep per window for the chosen detector only.
            #
            # pr_auc stays on `prauc`: a different (point-adjusted) metric, and
            # moving it would shift results for an unrelated reason.
            _, _, f1, _range_pr_auc, _ = range_based_precision_recall_f1_auc(y_true, y_scores)

            pr_auc = prauc(y_true, y_scores)
            # One vus_win for every window, so rewards stay on one scale.
            vus = (vus_score(y_scores, y_true, vus_win)
                   if 'vus' in metrics_required(metrics) and vus_win is not None
                   else float('nan'))
            reward = combine_metrics(metrics, {'f1': f1, 'pr_auc': pr_auc, 'vus': vus})
            if np.isnan(reward):
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
            # Norms, not the arrays. Printing every mu and every d-by-d Sigma
            # here cost 628 lines PER WINDOW — 95% of a run's log, and enough to
            # push every stage banner out of the WebUI's 20,000-line ring buffer
            # before the run finished. ||mu||^2 is the ranking criterion itself,
            # so this line still tracks what the arrays were being read for.
            # logger.info(f"Means: {means}")
            # logger.info(f"Covariances: {covariances}")
            logger.info("Posterior norms: " + ", ".join(
                f"{_m}: ||mu||^2={float(np.dot(_mu.flatten(), _mu.flatten())):.6g}"
                for _m, _mu in means.items()))

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
        n_context_features = data_windows[0].shape[0] if data_windows else 0
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
            "n_channels": n_context_features,
            # Per-window pre-update means; SHAP uses means_history[t] with
            # all_contexts[t] so every attribution reflects the beliefs held
            # when that window was decided.
            "means_history": _pre_means_hist,
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
        ax.plot(range(len(history)), smoothed_scores,
                label=model_name, linewidth=1.4)

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
        # SHORTENED: up to 107 legend entries stacked beside the axes, where a
        # full name pushes the legend over the plot. `draw_abbreviation_key`
        # below says what the short form stands for.
        ax.plot(range(T), series, label=abbreviate_detector(model_name),
                linewidth=1.4, color=colour_map[model_name])

    # Shade each regime, mark every boundary, and label every regime with the
    # model that dominates it — written vertically, centred in the regime span.
    if T > 0 and model_names:
        segments = reconstruct_regime_segments(regime_shifts, T, fallback_model=model_names[0])
        y_top = ax.get_ylim()[1]
        for (start, end, rm, _duration) in segments:
            if rm in colour_map:
                ax.axvspan(start, end + 1, alpha=0.08, color=colour_map[rm], lw=0)
            center = (start + end + 1) / 2.0
            # Shortened to match the legend, so one detector is not spelled two
            # ways inside one figure.
            ax.text(center, y_top * 0.97, abbreviate_detector(rm), fontsize=8,
                    ha='center', va='top', rotation=90, fontweight='bold',
                    alpha=0.85)
        for shift in regime_shifts:
            ax.axvline(x=shift['window'], color='black', linestyle='--', linewidth=0.9, alpha=0.7)

    title_suffix = ' (smoothed)' if smooth else ''
    ax.set_xlabel('Window')
    ax.set_ylabel('Expected Reward (mu_k^T * x_t)')
    ax.set_title('Expected Reward Trajectories Over Windows' + title_suffix)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    # One column, matching plot_ranking_score_trace: two columns of up to 107
    # detectors is wider than the axes it sits beside, and the eye has to track
    # a colour across a gap to use it. Font one step down so the single column
    # still fits the figure's height.
    ax.legend(loc='upper left', ncol=1, frameon=False, fontsize=8,
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    # One legend entry per detector, up to 107 of them, so the short form stays
    # and the key below says what it stands for.
    draw_abbreviation_key(fig, model_names)
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
    means_per_context: Optional[List[Dict[str, np.ndarray]]] = None,
) -> List[str]:
    """
    Return the top-k models by expected reward mu·x averaged over the given
    contexts.

    means_per_context : the beliefs held at each context, aligned with
        `contexts`. Supplying it makes the selection read the same mu·x the run
        actually saw at each window — which is what the regime prose ranks by
        (pre_expected_rewards_history) and what the plotted bars are computed
        from. Falling back to the final `means` here would let a plot show three
        detectors while the sentence beside it named a different runner-up.
    """
    if not contexts or not means:
        return list(means.keys())[:k]
    totals: Dict[str, float] = {m: 0.0 for m in means}
    for i, ctx in enumerate(contexts):
        at = means
        if means_per_context and i < len(means_per_context) and means_per_context[i]:
            at = {m: np.asarray(v).reshape(-1, 1)
                  for m, v in means_per_context[i].items()}
        for m, v in compute_expected_rewards(at, ctx).items():
            if m in totals:
                totals[m] += v
    ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:k]]


def _per_context_feature_shap_map(
    means: Dict[str, np.ndarray],
    top_models: List[str],
    context: np.ndarray,
    baseline: np.ndarray,
    n_context_features: int,
) -> Dict[str, np.ndarray]:
    """Per-context-feature SHAP for each model at a single context (raw, signed)."""
    return {
        m: aggregate_shap_per_context_feature(
            compute_shap_values(means[m].flatten(), context, baseline), n_context_features)
        for m in top_models
    }


def _fresh_plot_dir(directory: str) -> str:
    """
    Create `directory` and drop the .png files a previous run left there.

    Plot filenames embed the window range (e.g. regime_01_w0-4_NN_3.png), so a
    run with different regime boundaries writes new names and the stale ones
    linger — leaving several runs' plots side by side in one folder. Only .png
    files directly inside this generated directory are removed.
    """
    os.makedirs(directory, exist_ok=True)
    try:
        for name in os.listdir(directory):
            if name.lower().endswith('.png'):
                path = os.path.join(directory, name)
                if os.path.isfile(path):
                    os.remove(path)
    except OSError as e:
        logger.warning(f"Could not clear stale plots in {directory}: {e}")
    return directory


def _avg_per_context_feature_shap_map(
    means: Dict[str, np.ndarray],
    top_models: List[str],
    contexts: List[np.ndarray],
    baseline: np.ndarray,
    n_context_features: int,
    absolute: bool = False,
    means_per_context: Optional[List[Dict[str, np.ndarray]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Per-context-feature SHAP for each model, averaged over a list of contexts.

    absolute=False : raw signed average (meaningful for a regime, whose mean
                     context differs from the global baseline).
    absolute=True  : mean of |per-context-feature SHAP| — the standard SHAP global-
                     importance measure. Required for the whole-data average,
                     where the raw signed average is identically zero because
                     the baseline IS the mean of all contexts.

    means_per_context : the pre-update means for each context, index-aligned with
        `contexts`. When given, context t is explained with the beliefs held at
        window t. SHAP is bilinear in (mu, x), so the average of the per-window
        attributions is NOT the attribution of the averaged mean at the averaged
        context — the per-window loop below is the correct order.
    """
    out: Dict[str, np.ndarray] = {}
    n = max(len(contexts), 1)
    for m in top_models:
        mu_fixed = means[m].flatten() if means_per_context is None else None
        acc = np.zeros(n_context_features)
        for i, ctx in enumerate(contexts):
            if mu_fixed is not None:
                mu = mu_fixed
            else:
                if i >= len(means_per_context) or m not in means_per_context[i]:
                    continue
                mu = np.asarray(means_per_context[i][m]).flatten()
            pc = aggregate_shap_per_context_feature(compute_shap_values(mu, ctx, baseline), n_context_features)
            acc += np.abs(pc) if absolute else pc
        out[m] = acc / n
    return out


def _render_shap_comparison(
    per_context_feature_by_model: Dict[str, np.ndarray],
    top_models: List[str],
    top_n_context_features: int,
    title: str,
    save_path: str,
    ylabel: str = 'Per-context-feature SHAP contribution',
    n_context_features_total: Optional[int] = None,
    note: Optional[str] = None,
) -> None:
    """
    Shared grouped-bar renderer for per-context-feature comparison plots. Context features shown
    are the union of each model's top_n_context_features by |per-context-feature value|. One bar
    per model per context feature; legend placed outside the plot area.

    `ylabel` names the quantity: the SHAP stage leaves the default, the ranking
    stage passes the ||mu||^2 wording. Everything else about the two is identical,
    so they share this renderer rather than a near-copy of it.

    A footnote stating the context feature-selection rule is added automatically, and
    `note` is prepended to it. Every one of these figures shows a SUBSET of the
    context features — datasets here carry 9 to 38 — and a reader cannot tell from the
    bars alone whether an absent context feature was small or simply not plotted. The
    rule is generated here rather than written out at each call site so it can
    never drift from the selection the code just performed.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    candidate_context_features: set = set()
    for m in top_models:
        per_context_feature = per_context_feature_by_model[m]
        order = np.argsort(np.abs(per_context_feature))[::-1][:top_n_context_features]
        candidate_context_features.update(int(c) for c in order)
    selected = sorted(candidate_context_features)
    if not selected:
        return

    n_models = len(top_models)
    bar_width = 0.8 / max(n_models, 1)
    x_base = np.arange(len(selected))
    colour_map = {name: plt.cm.tab20(i / max(n_models, 1)) for i, name in enumerate(top_models)}

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(selected) + 4), 5))
    for i, m in enumerate(top_models):
        vals = per_context_feature_by_model[m][selected]
        ax.bar(x_base + i * bar_width, vals, bar_width,
               label=m, color=colour_map[m])

    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks(x_base + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels([f"cf{c}" for c in selected], rotation=45, ha='right')
    ax.set_xlabel('Context feature')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(loc='upper left', frameon=False, bbox_to_anchor=(1.01, 1), borderaxespad=0)

    scope = (f"{len(selected)} of {n_context_features_total}" if n_context_features_total
             else f"{len(selected)}")
    rule = (f"Context features shown ({scope}): the union over the plotted detectors of "
            f"each one's {top_n_context_features} largest |values|. A context feature absent "
            f"here was outside every plotted detector's top {top_n_context_features}, "
            f"not necessarily zero.")
    ax.text(0.0, -0.17, ((note + "  ") if note else "") + rule,
            transform=ax.transAxes, fontsize=7.5, color='dimgrey',
            va='top', ha='left', wrap=True)

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
    top_n_context_features: int = 10,
) -> None:
    """
    For the top_k_models (by ||mu||^2), draw horizontal bar charts of the
    top_n_context_features with the largest |per-context-feature SHAP| contribution at the
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
    n_context_features = shap_payload["n_channels"]

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
        per_context_feature = aggregate_shap_per_context_feature(shap_vals, n_context_features)
        e_r = float(np.dot(mu, context))

        order = np.argsort(np.abs(per_context_feature))[::-1][:top_n_context_features]
        vals = per_context_feature[order]
        labels = [f"cf{c}" for c in order]
        colours = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]

        y_pos = np.arange(len(order))
        ax.barh(y_pos, vals, color=colours)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=0.6)
        ax.set_xlabel('Per-context-feature SHAP contribution')
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
    top_n_context_features: int = 9,
    all_models: bool = False,
) -> None:
    """
    Grouped bar chart comparing models on the context features most relevant to their
    disagreement at the last window. Context features shown are the union of each model's
    top_n_context_features by |per-context-feature SHAP|.

    all_models : bool
        When False (default) the top_k_models by ||mu||^2 are shown, saved as
        shap_comparison_{iterations}.png. When True every model is shown, saved
        as shap_comparison_all_{iterations}.png.
    """
    if not shap_payload or shap_payload.get("n_channels", 0) <= 0:
        return

    context = shap_payload["explanation_context"]
    baseline = shap_payload["baseline_context"]
    n_context_features = shap_payload["n_channels"]

    if all_models:
        sel_models = _top_k_models_by_norm(means, len(means))
        suffix, title = '_all', 'SHAP Comparison Across All Models (at last window)'
    else:
        sel_models = _top_k_models_by_norm(means, top_k_models)
        suffix, title = '', 'SHAP Comparison Across Top Models (at last window)'
    if not sel_models:
        return

    per_context_feature = _per_context_feature_shap_map(means, sel_models, context, baseline, n_context_features)
    _render_shap_comparison(
        per_context_feature, sel_models, top_n_context_features,
        title=title,
        save_path=f'myresults/Thomposon/{dataset}/{entity}/shap_comparison{suffix}_{iterations}.png',
        n_context_features_total=n_context_features,
    )


_REWARD_YLABEL = r'Contribution to expected reward  $\mu^\top x$'


def _plot_per_regime(
    means: Dict[str, np.ndarray],
    shap_payload: Dict,
    regime_shifts: List[Dict],
    dataset: str,
    entity: str,
    iterations: int,
    *,
    stem: str,
    title_prefix: str,
    per_context_feature_fn,
    note_fn,
    ylabel: Optional[str] = None,
    top_k_models: int = 3,
    top_n_context_features: int = 9,
    all_models: bool = False,
) -> None:
    """One grouped-bar plot per leadership regime, for ONE quantity.

    Shared by the SHAP and expected-reward sets, which differ only in which
    per-context-feature map they build, what they call it, and the footnote. Everything
    that has to agree between them — the regime segmentation, the 0-based index
    matching the report's "Regime 0" and the IR's `ts.regime.0.*` ids, and the
    `regime_{NN}_w{start}-{end}_{leader}.png` filename the WebUI joiner parses
    back — is written once here, so the two sets cannot drift apart and stop
    pairing with the same regime sentence.

    `per_context_feature_fn(sel_models, regime_ctx, regime_mu)` returns the map to plot;
    `note_fn(n_windows)` returns the footnote.
    """
    if not shap_payload or shap_payload.get("n_channels", 0) <= 0:
        return
    contexts = shap_payload.get("all_contexts", [])
    if not contexts or not means:
        return
    n_context_features = shap_payload["n_channels"]

    fallback = _top_k_models_by_norm(means, 1)[0]
    segments = reconstruct_regime_segments(regime_shifts, len(contexts),
                                           fallback_model=fallback)
    every_model = _top_k_models_by_norm(means, len(means)) if all_models else None
    folder = f'{stem}_all_{iterations}' if all_models else f'{stem}_{iterations}'
    directory = _fresh_plot_dir(f'myresults/Thomposon/{dataset}/{entity}/{folder}/')
    mu_hist = shap_payload.get("means_history") or []
    for i, (start, end, model, _duration) in enumerate(segments):
        regime_ctx = contexts[start:end + 1]
        if not regime_ctx:
            continue
        regime_mu = mu_hist[start:end + 1] if mu_hist else None
        if all_models:
            sel_models = every_model
            scope = 'all models'
        else:
            sel_models = _top_k_models_by_expected_reward(
                means, regime_ctx, top_k_models, means_per_context=regime_mu)
            scope = f'top {top_k_models} by E[R] in regime'
        _render_shap_comparison(
            per_context_feature_fn(sel_models, regime_ctx, regime_mu),
            sel_models, top_n_context_features,
            title=(f'{title_prefix} — regime {i} ({model}, '
                   f'windows {start}-{end}, {scope})'),
            save_path=os.path.join(directory, f'regime_{i:02d}_w{start}-{end}_{model}.png'),
            ylabel=ylabel or 'Per-context-feature SHAP contribution',
            n_context_features_total=n_context_features,
            note=note_fn(end - start + 1),
        )


def plot_shap_per_regime(
    means: Dict[str, np.ndarray],
    shap_payload: Dict,
    regime_shifts: List[Dict],
    dataset: str,
    entity: str,
    iterations: int,
    top_k_models: int = 3,
    top_n_context_features: int = 9,
    all_models: bool = False,
) -> None:
    """
    One SHAP comparison plot per regime, showing the raw signed per-context-feature SHAP
    averaged over that regime's windows. A regime's mean context differs from the
    global baseline, so the signed average is meaningful (and direction-bearing).

    all_models : bool
        When False (default) each regime's plot shows the top_k_models by
        expected reward (mu·x) averaged over that regime's windows; saved under
        shap_per_regime_{iterations}/. When True every model is shown; saved
        under shap_per_regime_all_{iterations}/.
    """
    _plot_per_regime(
        means, shap_payload, regime_shifts, dataset, entity, iterations,
        stem='shap_per_regime', title_prefix='SHAP',
        per_context_feature_fn=lambda sel, ctx, mu: _avg_per_context_feature_shap_map(
            means, sel, ctx, shap_payload["baseline_context"],
            shap_payload["n_channels"], absolute=False, means_per_context=mu),
        note_fn=lambda n: f'SHAP averaged over the {n} windows of this regime.',
        top_k_models=top_k_models, top_n_context_features=top_n_context_features,
        all_models=all_models)


def plot_reward_per_regime(
    means: Dict[str, np.ndarray],
    shap_payload: Dict,
    regime_shifts: List[Dict],
    dataset: str,
    entity: str,
    iterations: int,
    top_k_models: int = 3,
    top_n_context_features: int = 9,
    all_models: bool = False,
) -> None:
    """
    Per-context-feature expected-reward contribution averaged over each regime's
    windows — the sibling of plot_shap_per_regime, and the default view beside
    the regime prose.

    Filenames match the SHAP set exactly (0-based index, window range, leader)
    so one joiner pairs either set with the same regime sentence.
    """
    _plot_per_regime(
        means, shap_payload, regime_shifts, dataset, entity, iterations,
        stem='reward_per_regime', title_prefix='Expected-reward contribution',
        per_context_feature_fn=lambda sel, ctx, mu: _avg_per_context_feature_reward_map(
            means, sel, ctx, shap_payload["n_channels"], means_per_context=mu),
        note_fn=lambda n: (f"Averaged over the {n} windows of this regime; "
                           f"each detector's bars sum to its mean expected "
                           f"reward here."),
        ylabel=_REWARD_YLABEL,
        top_k_models=top_k_models, top_n_context_features=top_n_context_features,
        all_models=all_models)



def plot_reward_average_all(
    means: Dict[str, np.ndarray],
    shap_payload: Dict,
    dataset: str,
    entity: str,
    iterations: int,
    top_k_models: int = 3,
    top_n_context_features: int = 9,
    all_models: bool = True,
) -> None:
    """
    One figure for the whole run: each context feature's mean contribution to a
    detector's expected reward, averaged over every window.

    This replaces mean|SHAP| as the run-level summary. The signed SHAP average
    over all windows is identically zero — the baseline IS the mean of those
    windows — which forced the old figure onto absolute values, and mean|SHAP|
    measures how much a context feature's influence VARIES, not how much it contributes.
    This average has no such defect: it is signed, non-degenerate, and its bars
    sum to the detector's expected reward on a typical window.
    """
    if not shap_payload or shap_payload.get("n_channels", 0) <= 0:
        return
    contexts = shap_payload.get("all_contexts", [])
    if not contexts or not means:
        return
    n_context_features = shap_payload["n_channels"]

    if all_models:
        sel_models = _top_k_models_by_norm(means, len(means))
        suffix, scope = 'all', 'all models'
    else:
        sel_models = _top_k_models_by_expected_reward(means, contexts, top_k_models)
        suffix, scope = f'top{top_k_models}', f'top {top_k_models} by E[R]'

    per_context_feature = _avg_per_context_feature_reward_map(
        means, sel_models, contexts, n_context_features,
        means_per_context=shap_payload.get("means_history") or None)
    _render_shap_comparison(
        per_context_feature, sel_models, top_n_context_features,
        title=f'Mean expected-reward contribution across all windows — {scope}',
        save_path=(f'myresults/Thomposon/{dataset}/{entity}/'
                   f'reward_average_{suffix}_{iterations}.png'),
        ylabel=_REWARD_YLABEL,
        n_context_features_total=n_context_features,
        note=("Averaged over every window; each detector's bars sum to its "
              "expected reward on a typical window."),
    )


def plot_shap_average_all(
    means: Dict[str, np.ndarray],
    shap_payload: Dict,
    dataset: str,
    entity: str,
    iterations: int,
    top_k_models: int = 3,
    top_n_context_features: int = 9,
    all_models: bool = True,
) -> None:
    """
    A single SHAP comparison plot summarising the whole run: the mean of
    |per-context-feature SHAP| across all windows (the standard SHAP global-importance
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
    n_context_features = shap_payload["n_channels"]

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

    per_context_feature = _avg_per_context_feature_shap_map(
        means, sel_models, contexts, baseline, n_context_features, absolute=True,
        means_per_context=shap_payload.get("means_history") or None)
    _render_shap_comparison(
        per_context_feature, sel_models, top_n_context_features,
        title=title,
        save_path=f'myresults/Thomposon/{dataset}/{entity}/shap_average_{suffix}_{iterations}.png',
        n_context_features_total=n_context_features,
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
    # pre-update beliefs), the leader's SHAP context features on the regime-aggregated
    # context, and the leader-vs-runner-up preference decomposition. Computed
    # ONCE here and consumed by BOTH the report below and the Intermediate
    # Representation, so the two always match.
    regimes_data: List[Dict[str, Any]] = []
    try:
        all_ctx = (shap_payload or {}).get("all_contexts") or []
        base_ctx = (shap_payload or {}).get("baseline_context")
        n_ch = int((shap_payload or {}).get("n_channels", 0) or 0)
        # Pre-update means per window; empty falls back to the final means.
        mu_hist = (shap_payload or {}).get("means_history") or []
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

            # Context feature contributions are split by SIGN before truncation: a
            # magnitude-sorted signed list invites "driven by" phrasings whose
            # top entries actually push the other way. Top-3 per direction.
            def _split_by_sign(vals: np.ndarray) -> Tuple[list, list]:
                pos = [(int(c), float(vals[c])) for c in np.argsort(vals)[::-1]
                       if vals[c] > 0][:3]
                neg = [(int(c), float(vals[c])) for c in np.argsort(vals)
                       if vals[c] < 0][:3]
                return pos, neg

            shap_raising = shap_lowering = None
            reward_raising = reward_lowering = None
            pref_favor_leader = pref_favor_runner = None
            edge_favor_leader = edge_favor_runner = None
            pref_gap = float('nan')
            edge_gap = float('nan')
            if leader is not None and n_ch > 0 and len(all_ctx) > seg_end:
                # Per-window attribution with the beliefs held at that window,
                # then averaged over the regime. SHAP is bilinear in (mu, x), so
                # this is NOT the same as explaining the averaged mean at the
                # averaged context — and only this order makes the context feature
                # deltas sum to the regime's reported mean-reward gap.
                win = range(seg_start, seg_end + 1)
                mu_at = (lambda t, m: np.asarray(mu_hist[t][m]).flatten()
                         if t < len(mu_hist) and m in mu_hist[t]
                         else means[m].flatten())
                n_win = max(len(win), 1)
                pc_l = np.zeros(n_ch)
                pc_r = np.zeros(n_ch)
                rc_l = np.zeros(n_ch)          # raw expected-reward contribution
                rc_r = np.zeros(n_ch)
                gap_acc = 0.0
                edge_acc = 0.0
                have_runner = runner is not None and runner in means
                for t in win:
                    ctx_t = np.asarray(all_ctx[t], dtype=float)
                    mu_l = mu_at(t, leader)
                    pc_l += aggregate_shap_per_context_feature(
                        compute_shap_values(mu_l, ctx_t, base_ctx), n_ch)
                    rc_l += reward_contribution_per_context_feature(mu_l, ctx_t, n_ch)
                    if have_runner:
                        mu_r = mu_at(t, runner)
                        pc_r += aggregate_shap_per_context_feature(
                            compute_shap_values(mu_r, ctx_t, base_ctx), n_ch)
                        rc_r += reward_contribution_per_context_feature(mu_r, ctx_t, n_ch)
                        gap_acc += float(np.dot(mu_l - mu_r, ctx_t - base_ctx))
                        edge_acc += float(np.dot(mu_l - mu_r, ctx_t))
                pc_l /= n_win
                rc_l /= n_win
                # What the leader's expected reward is actually MADE OF here.
                # These sum to its mean expected reward over the regime, which
                # the SHAP split cannot claim — it drops the mu.baseline term.
                reward_raising, reward_lowering = _split_by_sign(rc_l)
                shap_raising, shap_lowering = _split_by_sign(pc_l)
                if have_runner:
                    pc_r /= n_win
                    rc_r /= n_win
                    # The narrated edge: the leader's own expected-reward split
                    # minus the runner-up's. These deltas sum to the gap in
                    # expected reward the regime is actually decided by — the
                    # same quantity `reward_gap` reports — so the sentence's
                    # context feature and its headline number describe one thing.
                    edge_gap = edge_acc / n_win
                    edge_favor_leader, edge_favor_runner = _split_by_sign(rc_l - rc_r)
                    delta = pc_l - pc_r
                    # Baseline-relative, so the context feature deltas sum to it exactly.
                    pref_gap = gap_acc / n_win
                    pref_favor_leader, pref_favor_runner = _split_by_sign(delta)

            regimes_data.append({
                "index": seg_idx, "start": seg_start, "end": seg_end,
                "duration": int(seg_dur), "leader": leader,
                "rewards_top": top3, "reward_gap": gap, "runner_up": runner,
                # The narrated context features: what the leader's expected reward is
                # made of here. SHAP's split rides along for the deviation
                # clause and the alternate plot, but no longer leads.
                "reward_raising": reward_raising, "reward_lowering": reward_lowering,
                "shap_raising": shap_raising, "shap_lowering": shap_lowering,
                # The leader's edge over the runner-up in the SAME units as the
                # clause before it — a slice of expected reward, not a
                # baseline-relative deviation.
                "edge_favor_leader": edge_favor_leader,
                "edge_favor_runner": edge_favor_runner,
                "edge_gap": edge_gap,
                # The SHAP version of the same comparison, kept machine-readable
                # for the alternate per-regime plot; it no longer feeds the prose.
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

        f.write("\n--- Per-Regime Expected Rewards & Context-Feature Attribution ---\n")
        f.write("(Mean E[reward] over each regime's windows from the recorded pre-update\n")
        f.write(" beliefs. Two context feature splits, both averaged over the regime's windows\n")
        f.write(" using the beliefs held at each one. CONTRIBUTION is the raw split of\n")
        f.write(" mu.x, whose parts sum to E[reward]; DEVIATION is the SHAP split, which\n")
        f.write(" measures departure from the run's average window and sums to nothing\n")
        f.write(" a reader can name. Matches the Intermediate Representation.)\n")
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
            if r["reward_raising"] or r["reward_lowering"]:
                raise_s = ", ".join(f"ch {c} {v:+.4f}"
                                    for c, v in (r["reward_raising"] or [])) or "none"
                lower_s = ", ".join(f"ch {c} {v:+.4f}"
                                    for c, v in (r["reward_lowering"] or [])) or "none"
                f.write(f"  CONTRIBUTION — context features supplying {r['leader']}'s "
                        f"E[reward]: {raise_s}\n")
                f.write(f"  CONTRIBUTION — context features reducing it: {lower_s}\n")
            if r["shap_raising"] or r["shap_lowering"]:
                raise_s = ", ".join(f"ch {c} {v:+.4f}"
                                    for c, v in (r["shap_raising"] or [])) or "none"
                lower_s = ", ".join(f"ch {c} {v:+.4f}"
                                    for c, v in (r["shap_lowering"] or [])) or "none"
                f.write(f"  DEVIATION — context features above their usual contribution: "
                        f"{raise_s}\n")
                f.write(f"  DEVIATION — context features below it: {lower_s}\n")
            # The narrated edge, in contribution units: these deltas sum to the
            # leader-vs-runner-up gap in expected reward reported above.
            has_edge = r.get("edge_favor_leader") or r.get("edge_favor_runner")
            if has_edge and r["runner_up"] and not np.isnan(r.get("edge_gap", float('nan'))):
                favored = r["leader"] if r["edge_gap"] >= 0 else r["runner_up"]
                fl = ", ".join(f"ch {c} {d:+.4f}"
                               for c, d in (r["edge_favor_leader"] or [])) or "none"
                fr = ", ".join(f"ch {c} {d:+.4f}"
                               for c, d in (r["edge_favor_runner"] or [])) or "none"
                f.write(f"  EDGE (contribution) {r['leader']} vs {r['runner_up']}: "
                        f"E[reward] favors {favored} by {abs(r['edge_gap']):.4f}\n")
                f.write(f"    context features favoring {r['leader']}: {fl}\n")
                f.write(f"    context features favoring {r['runner_up']}: {fr}\n")
            has_pref = r["pref_favor_leader"] or r["pref_favor_runner"]
            if has_pref and r["runner_up"] and not np.isnan(r["pref_gap"]):
                favored = r["leader"] if r["pref_gap"] >= 0 else r["runner_up"]
                fl = ", ".join(f"ch {c} {d:+.4f}"
                               for c, d in (r["pref_favor_leader"] or [])) or "none"
                fr = ", ".join(f"ch {c} {d:+.4f}"
                               for c, d in (r["pref_favor_runner"] or [])) or "none"
                f.write(f"  EDGE (deviation) {r['leader']} vs {r['runner_up']}: linear "
                        f"preference score at the regime-average context favors "
                        f"{favored} by {abs(r['pref_gap']):.4f}\n")
                f.write(f"    context features favoring {r['leader']}: {fl}\n")
                f.write(f"    context features favoring {r['runner_up']}: {fr}\n")

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
            f.write("Per-context-feature         = sum of phi_i over the context feature's window timesteps\n\n")

            top_models = _top_k_models_by_norm(means, 3)
            per_context_feature_by_model: Dict[str, np.ndarray] = {}
            for rank, model_name in enumerate(top_models, 1):
                mu = means[model_name].flatten()
                shap_vals = compute_shap_values(mu, ctx, base)
                per_ch = aggregate_shap_per_context_feature(shap_vals, n_ch)
                per_context_feature_by_model[model_name] = per_ch
                e_r = float(np.dot(mu, ctx))
                e_r_base = float(np.dot(mu, base))
                delta = e_r - e_r_base
                f.write(f"  {rank}. {model_name}  "
                        f"(E[R | last] = {e_r:+.4f},  baseline E[R] = {e_r_base:+.4f},  delta = {delta:+.4f})\n")
                f.write(f"     Top 5 context features by |per-context-feature SHAP|:\n")
                top_idx = np.argsort(np.abs(per_ch))[::-1][:5]
                for c in top_idx:
                    f.write(f"       context feature {int(c):>3} : {per_ch[c]:+.4f}\n")
                f.write(f"     Sum over all context features: {float(per_ch.sum()):+.4f}\n\n")

            if len(top_models) >= 2:
                top, second = top_models[0], top_models[1]
                gap = float(np.dot(means[top].flatten() - means[second].flatten(), ctx))
                delta_per_ch = per_context_feature_by_model[top] - per_context_feature_by_model[second]
                f.write("--- SHAP Preference Decomposition ---\n")
                f.write(f"Top model: {top}  vs  2nd: {second}\n")
                f.write(f"Preference gap at last window: (mu_{top} - mu_{second})^T x_last = {gap:+.4f}\n")
                f.write("Top 5 context features driving the preference:\n")
                top_idx = np.argsort(np.abs(delta_per_ch))[::-1][:5]
                for c in top_idx:
                    a = per_context_feature_by_model[top][c]
                    b = per_context_feature_by_model[second][c]
                    f.write(f"  context feature {int(c):>3} : "
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
        ir_doc = ir.build_thompson_ir(
            dataset, entity, n_windows=T,
            final_ranking=ranking,
            regimes=regimes_data,
            shifts=regime_shifts,
            blip_count=len(blip_windows),
            state_fractions={s: state_counts[s] / state_total for s in state_counts},
            # The tallies as well as the shares: the IR states both, and a share
            # rounded back to a window count can be one out on a long run.
            state_counts={s: int(state_counts[s]) for s in state_counts},
            final_state=selection_states[-1] if selection_states else "not_available",
            # Context feature names when the loader supplied them; the IR falls back to
            # "context feature N" for datasets whose sources have no column headers.
            context_feature_names=(shap_payload or {}).get("channel_names"),
            n_context_features=(shap_payload or {}).get("n_channels"),
        )
        ir.write_stage_ir(ir_doc, dataset, entity, "ir_thompson")
    except Exception as e:
        logger.error(f"Thompson IR emission failed (non-fatal): {e}")


# ── Ranking-criterion explainability (||mu_k||^2) ────────────────────────────
#
# The stage above explains mu^T x — the expected reward that drives per-window
# selection. Everything below explains the quantity the detectors are actually
# ranked by, mu^T mu, which is context-free and therefore decomposes over
# context features on its own, with no baseline and no SHAP.

def _norm_scores_at(means_at: Dict[str, np.ndarray]) -> Dict[str, float]:
    """||mu||^2 for every detector at one window."""
    return {m: float(np.dot(mu.flatten(), mu.flatten())) for m, mu in means_at.items()}


def _context_feature_pairs(values: np.ndarray, top_n: Optional[int] = None,
                   by_magnitude: bool = False) -> List[Tuple[int, float]]:
    """[(context_feature_index, value)] sorted for presentation, optionally truncated."""
    pairs = [(int(c), float(v)) for c, v in enumerate(values)]
    pairs.sort(key=lambda cv: -abs(cv[1]) if by_magnitude else -cv[1])
    return pairs[:top_n] if top_n else pairs


def _regime_ranking_facts(means_history: List[Dict[str, np.ndarray]],
                          segments: List[Tuple[int, int, str, int]],
                          n_context_features: int,
                          top_n_context_features: int = 3) -> List[Dict[str, Any]]:
    """
    Per-regime facts, read at the regime's LAST window — the state the leader
    had accumulated by the time it handed over, which is what its lead over that
    regime's runner-up rests on.

    The single source for both the report/IR and the per-regime plots, so the
    prose and the figure beside it can never disagree.
    """
    facts: List[Dict[str, Any]] = []
    for i, (start, end, leader, duration) in enumerate(segments):
        if end >= len(means_history):
            continue
        at = means_history[end]
        scores = _norm_scores_at(at)
        rivals = sorted(((s, m) for m, s in scores.items() if m != leader), reverse=True)
        runner = rivals[0][1] if rivals else None

        top_context_features = _context_feature_pairs(
            aggregate_squared_per_context_feature(at[leader], n_context_features), top_n_context_features)
        gap_context_features = []
        if runner is not None:
            gap_context_features = _context_feature_pairs(
                rank_gap_decomposition(at[leader], at[runner], n_context_features),
                top_n_context_features, by_magnitude=True)

        facts.append({
            "index": i, "start": int(start), "end": int(end),
            "duration": int(duration), "leader": leader, "runner_up": runner,
            "top_channels": top_context_features, "gap_channels": gap_context_features,
            "score": scores.get(leader),
            "runner_score": scores.get(runner) if runner else None,
        })
    return facts


def plot_ranking_criterion(means_history: List[Dict[str, np.ndarray]],
                           segments: List[Tuple[int, int, str, int]],
                           warmup: int, dataset: str, entity: str,
                           iterations: int) -> None:
    """
    ||mu_k||^2 for every detector across the run, with the leadership regimes
    shaded and the excluded warm-up greyed out.

    The sibling of plot_expected_rewards, on the axis the ranking actually uses:
    where that plot shows a detector's chance of being picked next, this one
    shows the quantity it is finally ranked by. Unsmoothed on purpose — regimes
    here are read off the raw series, so a smoothed curve would show boundaries
    the segmentation did not use.
    """
    if not means_history:
        return
    plt.rcParams.update({
        "font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    })
    model_names = sorted(means_history[0].keys())
    T = len(means_history)
    series = {m: [float(np.dot(h[m].flatten(), h[m].flatten())) for h in means_history]
              for m in model_names}

    fig, ax = plt.subplots(figsize=(12, 6))
    colour_map = {m: plt.cm.tab20(i / max(len(model_names), 1))
                  for i, m in enumerate(model_names)}
    for m in model_names:
        # SHORTENED, for the same reason as the expected-reward traces: one
        # legend entry per detector, outside the axes. The key underneath
        # says what each stands for.
        ax.plot(range(T), series[m], label=abbreviate_detector(m),
                linewidth=1.4, color=colour_map[m])

    # Fix the ceiling before annotating: every axvspan below would otherwise
    # move it, and the labels would drift off the top of the axes.
    ymax = ax.get_ylim()[1]
    ax.set_ylim(ax.get_ylim()[0], ymax * 1.12)
    ymax = ax.get_ylim()[1]

    if warmup > 0:
        ax.axvspan(-0.5, warmup - 0.5, color='grey', alpha=0.18, zorder=0)
        ax.text(max(warmup / 2.0 - 0.5, 0.0), ymax * 0.5, 'warm-up',
                ha='center', va='center', rotation=90, fontsize=8,
                color='dimgrey')
    for i, (start, end, leader, _duration) in enumerate(segments):
        if i:
            ax.axvline(start - 0.5, color='black', linestyle='--', linewidth=0.7,
                       alpha=0.5)
        ax.axvspan(start - 0.5, end + 0.5, color=colour_map.get(leader, 'grey'),
                   alpha=0.10, zorder=0)
        # Alternate the label height so neighbouring short regimes — which pure
        # run-length encoding does produce wherever two detectors are near-tied —
        # do not print on top of each other.
        # Shortened to match the legend: a figure that spelled the leader out
        # here and abbreviated the same detector three inches to the right
        # would read as two different detectors.
        ax.text((start + end) / 2.0, ymax * (0.99 if i % 2 == 0 else 0.90),
                f'R{i} {abbreviate_detector(leader)}', ha='center', va='top',
                fontsize=7, rotation=90, color='black', alpha=0.75)

    ax.set_xlabel('Window')
    ax.set_ylabel(r'Ranking score  $\|\mu_k\|^2$')
    ax.set_title('Ranking score over the run, shaded by leadership regime')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(loc='upper left', frameon=False, bbox_to_anchor=(1.01, 1),
              borderaxespad=0)
    # Short names kept: they label both the legend and every regime band, and a
    # regime band is only as wide as the regime.
    draw_abbreviation_key(fig, list(series))

    directory = f'myresults/Thomposon/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    plt.tight_layout(pad=1.2)
    plt.savefig(os.path.join(directory, f'ranking_criterion_{iterations}.png'),
                format='png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_ranking_final(means: Dict[str, np.ndarray],
                       list_of_chosen_models: List[str],
                       dataset: str, entity: str, iterations: int) -> None:
    """
    The final ranking as a bar chart, each bar annotated with how many windows
    that detector was actually selected in.

    The selection count is on the plot because it is the ranking's main
    confound: mu only moves in the windows where its arm was pulled, so a low
    bar on two selections means "barely tried", not "tried and found wanting".
    """
    if not means:
        return
    plt.rcParams.update({
        "font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    })
    counts: Dict[str, int] = {}
    for m in list_of_chosen_models or []:
        counts[m] = counts.get(m, 0) + 1

    ranking = sorted(((float(np.dot(mu.flatten(), mu.flatten())), m)
                      for m, mu in means.items()), reverse=True)
    labels = [m for _s, m in ranking][::-1]
    values = [s for s, _m in ranking][::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(labels) + 1.5)))
    colours = ['#3B5BDB' if i == len(labels) - 1 else '#AAB2C8'
               for i in range(len(labels))]
    # Ticks carry the long name; `labels` stays canonical below, since it is
    # what `counts` is keyed by.
    bars = ax.barh(labels, values, color=colours)
    span = max(values) if values else 1.0
    for bar, name in zip(bars, labels):
        ax.text(bar.get_width() + span * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.6f}  ({counts.get(name, 0)} selections)',
                va='center', fontsize=8)

    ax.set_xlim(0, span * 1.35 if span else 1.0)
    ax.set_xlabel(r'Ranking score  $\|\mu_k\|^2$')
    ax.set_title('Final ranking, with the number of windows each detector was tried')
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.6)

    directory = f'myresults/Thomposon/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    plt.tight_layout(pad=1.2)
    plt.savefig(os.path.join(directory, f'ranking_final_{iterations}.png'),
                format='png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_ranking_channels(means: Dict[str, np.ndarray], n_context_features: int,
                          dataset: str, entity: str, iterations: int,
                          top_k_models: int = 3, top_n_context_features: int = 9,
                          all_models: bool = False) -> None:
    """Per-context-feature split of the final ||mu||^2, top-k detectors or all of them."""
    if not means or n_context_features <= 0:
        return
    models = (_top_k_models_by_norm(means, len(means)) if all_models
              else _top_k_models_by_norm(means, top_k_models))
    per_context_feature = {m: aggregate_squared_per_context_feature(means[m], n_context_features)
                   for m in models}
    suffix = '_all' if all_models else ''
    scope = 'all models' if all_models else f'top {top_k_models} by score'
    directory = f'myresults/Thomposon/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    _render_shap_comparison(
        per_context_feature, models, top_n_context_features,
        title=f'Where each detector\'s ranking score comes from ({scope})',
        save_path=os.path.join(directory,
                               f'ranking_channels{suffix}_{iterations}.png'),
        ylabel=r'Contribution to $\|\mu_k\|^2$',
        n_context_features_total=n_context_features,
        note='Final weights, at the end of the run.',
    )


def plot_ranking_gap(means: Dict[str, np.ndarray], n_context_features: int,
                     dataset: str, entity: str, iterations: int,
                     top_n_context_features: int = 12) -> None:
    """
    The winner's margin over the runner-up, split per context feature and signed.

    This is the plot that answers the ranking question directly: the bars sum
    exactly to the gap between the two scores, so a green bar is a context feature that
    put the winner ahead and a red one is a context feature the runner-up won.
    """
    if not means or n_context_features <= 0 or len(means) < 2:
        return
    order = _top_k_models_by_norm(means, 2)
    winner, runner = order[0], order[1]
    gap = rank_gap_decomposition(means[winner], means[runner], n_context_features)
    pairs = _context_feature_pairs(gap, top_n_context_features, by_magnitude=True)
    if not pairs:
        return
    pairs = sorted(pairs, key=lambda cv: cv[1])

    plt.rcParams.update({
        "font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(pairs) + 1.5)))
    ax.barh([f'cf{c}' for c, _v in pairs], [v for _c, v in pairs],
            color=['#2F9E44' if v >= 0 else '#C92A2A' for _c, v in pairs])
    ax.axvline(0, color='black', linewidth=0.7)
    total = float(np.sum(gap))
    ax.set_xlabel(r'Contribution to the gap in $\|\mu\|^2$')
    ax.set_title(f'{winner} vs {runner}: where the {total:+.6f} margin came from\n'
                 f'(green: {winner} ahead, red: {runner} ahead)')
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.6)

    directory = f'myresults/Thomposon/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    plt.tight_layout(pad=1.2)
    plt.savefig(os.path.join(directory, f'ranking_gap_{iterations}.png'),
                format='png', dpi=300, bbox_inches='tight')
    plt.close()


# ── Per-window context feature aggregates, for on-demand rendering ──────────────────
# The three per-window families above (shap_, reward_, ranking_) each wrote one
# PNG per window in three scopes — nine folders, ~1,100 frames and 167 MB for a
# single 173-window entity — of which a reader opens a handful. What every one
# of those frames draws is a per-model per-context-feature vector, and that is three
# orders of magnitude smaller than its own rendering: 173x11x9x3 floats is half
# a megabyte against 167 MB of PNG.
#
# So the run persists the numbers and the WebUI draws the frame that is asked
# for (WebUI/ondemand.render_per_window). Same trade as the ranking-gap pair
# picker, and it makes the `_all` and `_every10` sets free: they were never
# different figures, only a different top-k and a different stride, which are
# arguments to a renderer rather than folders on disk.
#
# The plot_*_per_window functions above are kept — they are the reference
# rendering, and a thesis figure can still be minted from one directly — but the
# pipeline no longer calls them.

PER_WINDOW_SCHEMA = 1

# Carried in the file itself so the on-demand renderer cannot drift from the
# producer: these are the titles, axis labels and footnotes the eager functions
# above passed to _render_shap_comparison, with {t} for the window index and {k}
# for the top-k. `rank_by` names the quantity each set's top-k is chosen on —
# every one of them is an exact sum of a stored row, so the selection is
# reproducible from this file alone and cannot disagree with the bars.
_PER_WINDOW_KINDS: Dict[str, Dict[str, Any]] = {
    "reward": {
        "label": "Reward contribution",
        "ylabel": _REWARD_YLABEL,
        "title_top": "Expected-reward contribution — window {t} (top {k} by E[R] in window)",
        "title_all": "Expected-reward contribution — window {t} (all models)",
        "note": ("Each detector's bars sum to its expected reward at window {t}; "
                 "no baseline is subtracted."),
        "rank_by": "reward",
        "all_by": "final",
    },
    "shap": {
        "label": "Deviation from a typical window",
        "ylabel": "Per-context-feature SHAP contribution",
        "title_top": "SHAP — window {t} (top {k} by E[R] in window)",
        "title_all": "SHAP — window {t} (all models)",
        "note": None,
        # The SHAP set's top-k was chosen on expected reward, not on SHAP, so
        # the frame shows the same detectors as its reward sibling.
        "rank_by": "reward",
        "all_by": "final",
    },
    "ranking": {
        "label": "Ranking score",
        "ylabel": r"Contribution to $\|\mu_k\|^2$",
        "title_top": "Ranking score by context feature — window {t} (top {k} by score at this window)",
        "title_all": "Ranking score by context feature — window {t} (all detectors)",
        "note": ("Weights as they stood at window {t}. The score is cumulative, "
                 "so these bars are the total accumulated up to this window."),
        "rank_by": "ranking",
        "all_by": "ranking",
    },
}


def _json_row(values: np.ndarray) -> List[Optional[float]]:
    """One context feature vector at 6 significant digits; non-finite becomes null.

    Six digits is far beyond what a bar chart can show and keeps the file an
    order of magnitude smaller than full repr. `null` rather than NaN because
    NaN is not valid JSON, and a window only produces one after the run logged
    an error for it.
    """
    return [None if not np.isfinite(v) else float(f"{float(v):.6g}")
            for v in np.asarray(values, dtype=float).ravel()]


def save_per_window_context_features(
    means: Dict[str, np.ndarray],
    shap_payload: Optional[Dict],
    dataset: str,
    entity: str,
    iterations: int,
) -> Optional[str]:
    """Persist every per-window frame's numbers as one JSON file.

    Returns the path written, or None when the run carries nothing to write
    (no explain payload, no context features, no windows) — the same conditions under
    which the plot functions returned without drawing anything.
    """
    if not shap_payload or not means:
        return None
    n_context_features = int(shap_payload.get("n_channels") or 0)
    contexts = shap_payload.get("all_contexts") or []
    mu_hist = shap_payload.get("means_history") or []
    baseline = shap_payload.get("baseline_context")
    if n_context_features <= 0 or not contexts or baseline is None:
        return None

    # Appended in the same block of the bandit loop, so these are index-aligned
    # by construction; min() is belt and braces for a run that errored out of
    # one of them.
    n_windows = min(len(contexts), len(mu_hist)) if mu_hist else len(contexts)
    if n_windows <= 0:
        return None

    # Registry order, matching the dict the top-k selectors iterate, so a tie
    # breaks the same way here as it did in the eager plots.
    models = list(means.keys())
    sets: Dict[str, List[List[List[Optional[float]]]]] = {"reward": [], "shap": [], "ranking": []}
    for t in range(n_windows):
        ctx = contexts[t]
        at = mu_hist[t] if t < len(mu_hist) else {m: means[m] for m in models}
        reward_frame, shap_frame, ranking_frame = [], [], []
        for m in models:
            mu = np.asarray(at[m]).flatten() if m in at else np.asarray(means[m]).flatten()
            reward_frame.append(_json_row(
                reward_contribution_per_context_feature(mu, ctx, n_context_features)))
            shap_frame.append(_json_row(aggregate_shap_per_context_feature(
                compute_shap_values(mu, ctx, baseline), n_context_features)))
            ranking_frame.append(_json_row(
                aggregate_squared_per_context_feature(mu, n_context_features)))
        sets["reward"].append(reward_frame)
        sets["shap"].append(shap_frame)
        sets["ranking"].append(ranking_frame)

    document = {
        "schema": PER_WINDOW_SCHEMA,
        "dataset": dataset,
        "entity": entity,
        "iterations": iterations,
        "n_channels": n_context_features,
        "n_windows": n_windows,
        "top_k_models": 3,
        "top_n_channels": 9,
        "models": models,
        # The order the `_all` frames of the reward and SHAP sets used: final
        # ||mu||^2, fixed for the whole run. The ranking set re-sorts per window.
        "models_by_final_norm": _top_k_models_by_norm(means, len(means)),
        "kinds": _PER_WINDOW_KINDS,
        "sets": sets,
    }

    directory = f'myresults/Thomposon/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'per_window_channels_{iterations}.json')
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, separators=(",", ":"))
    logger.info(f"Per-window context feature aggregates written to {path} "
                f"({n_windows} windows x {len(models)} detectors x {n_context_features} context features)")
    return path


def plot_ranking_per_regime(means_history: List[Dict[str, np.ndarray]],
                            segments: List[Tuple[int, int, str, int]],
                            n_context_features: int, dataset: str, entity: str,
                            iterations: int, top_n_context_features: int = 9) -> None:
    """
    One figure per leadership regime: the leader's per-context-feature score beside that
    regime's runner-up, read at the regime's last window.

    Filenames mirror shap_per_regime_{it}/ exactly — 0-based index, window range,
    leader — so the same WebUI joiner pairs each figure with its own sentence.
    """
    if not means_history or not segments or n_context_features <= 0:
        return
    directory = _fresh_plot_dir(
        f'myresults/Thomposon/{dataset}/{entity}/ranking_per_regime_{iterations}/')
    for fact in _regime_ranking_facts(means_history, segments, n_context_features,
                                      top_n_context_features=n_context_features):
        end, leader, runner = fact["end"], fact["leader"], fact["runner_up"]
        at = means_history[end]
        # Top three by ||mu||^2 at this window, which is the quantity the
        # regime itself is defined on — so the leader heads the list and the
        # runner-up follows. Showing a third gives the pair a scale: two bars
        # alone cannot say whether the runner-up was close to the leader or
        # merely the best of a distant field.
        models = _top_k_models_by_norm(at, 3)
        for m in ([leader] + ([runner] if runner else [])):
            if m and m in at and m not in models:
                models.append(m)
        per_context_feature = {m: aggregate_squared_per_context_feature(at[m], n_context_features)
                       for m in models}
        rng = f'{fact["start"]}-{end}'
        _render_shap_comparison(
            per_context_feature, models, top_n_context_features,
            title=(f'Ranking score by context feature — regime {fact["index"]} '
                   f'({leader}, windows {rng}), as at window {end}'),
            save_path=os.path.join(
                directory,
                f'regime_{fact["index"]:02d}_w{rng}_{leader}.png'),
            ylabel=r'Contribution to $\|\mu_k\|^2$',
            n_context_features_total=n_context_features,
            # Says plainly what a per-regime figure of a cumulative quantity can
            # and cannot mean. Unlike the SHAP stage's per-regime plot, which
            # averages a per-window attribution over the regime, this is a
            # single snapshot: ||mu||^2 only moves when the arm is selected and
            # never resets, so the bars carry everything the detector had
            # accumulated by then, not what this regime contributed.
            note=(f'Weights as they stood at window {end}, the last of this '
                  f'regime. The score is cumulative, so this is the state '
                  f'reached by the end of the regime, not the regime\'s own '
                  f'contribution.'),
        )


def explain_thompson_ranking(
    means: Dict[str, np.ndarray],
    list_of_chosen_models: List[str],
    shap_payload: Optional[Dict],
    segments: List[Tuple[int, int, str, int]],
    warmup_used: int,
    dataset: str,
    entity: str,
    iterations: int,
) -> None:
    """
    Write the ranking-criterion report and its IR.

    Answers "why did Thompson Sampling rank the detectors as it did", which the
    sibling stage does not: that one explains mu^T x, the per-window expected
    reward, while the ranking is by mu^T mu. Both decompositions used here are
    exact — the per-detector shares sum to that detector's score, and the
    winner-minus-runner-up terms sum to the margin — so nothing is approximated
    and there is no baseline to choose.
    """
    if not means:
        return
    payload = shap_payload or {}
    n_context_features = int(payload.get("n_channels") or 0)
    means_history: List[Dict[str, np.ndarray]] = payload.get("means_history") or []
    T = len(means_history)

    final_scores = {m: float(np.dot(mu.flatten(), mu.flatten())) for m, mu in means.items()}
    ranking = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)
    winner = ranking[0][0]
    runner_up = ranking[1][0] if len(ranking) > 1 else None

    winner_context_features = (_context_feature_pairs(aggregate_squared_per_context_feature(means[winner], n_context_features))
                       if n_context_features > 0 else [])
    gap_context_features = (_context_feature_pairs(
        rank_gap_decomposition(means[winner], means[runner_up], n_context_features),
        by_magnitude=True) if (runner_up and n_context_features > 0) else [])

    counts: Dict[str, int] = {m: 0 for m in means}
    for m in list_of_chosen_models or []:
        counts[m] = counts.get(m, 0) + 1

    regimes_data = _regime_ranking_facts(means_history, segments, n_context_features)

    directory = f'myresults/Thomposon/{dataset}/{entity}/'
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(directory, f'ranking_explainability_{iterations}.txt')
    with open(output_file, 'w') as f:
        f.write("Thompson Sampling — ranking criterion ||mu_k||^2\n")
        f.write("=" * 52 + "\n")
        f.write(f"Dataset: {dataset}   Entity: {entity}   Windows: {T}   "
                f"Context features: {n_context_features}\n")

        f.write("\n--- Final Ranking (by ||mu_k||^2) ---\n")
        f.write(f"  {'Rank':>4}  {'Model':>12}  {'Score':>12}  {'Selections':>11}\n")
        f.write("  " + "-" * 45 + "\n")
        for rank, (m, score) in enumerate(ranking, 1):
            f.write(f"  {rank:>4}  {m:>12}  {score:>12.6f}  {counts.get(m, 0):>11}\n")

        f.write(f"\n--- Per-Context-Feature Decomposition of {winner}'s Score ---\n")
        f.write("  Contributions are sums of squared weights: non-negative, and they\n"
                "  add up to the score exactly.\n")
        total = sum(v for _c, v in winner_context_features) or 1.0
        for c, v in winner_context_features[:20]:
            f.write(f"  context feature {c:>3}  {v:>12.6f}  ({100.0 * v / total:>5.1f}%)\n")
        f.write(f"  {'total':>11}  {sum(v for _c, v in winner_context_features):>12.6f}\n")

        if runner_up:
            f.write(f"\n--- {winner} vs {runner_up}: Gap Decomposition ---\n")
            f.write("  Signed, and sums to the difference between the two scores.\n")
            for c, v in gap_context_features[:20]:
                side = winner if v >= 0 else runner_up
                f.write(f"  context feature {c:>3}  {v:>+12.6f}  favours {side}\n")
            f.write(f"  {'gap':>11}  {sum(v for _c, v in gap_context_features):>+12.6f}\n")

        f.write("\n--- Leadership Regimes ---\n")
        f.write(f"  Warm-up windows excluded: {warmup_used}\n")
        if not regimes_data:
            f.write("  No regime could be formed.\n")
        for r in regimes_data:
            f.write(f"  Regime {r['index']:>2}  windows {r['start']:>4}-{r['end']:<4} "
                    f"({r['duration']:>3} windows)  leader {r['leader']:>12}"
                    f"  runner-up {str(r['runner_up']):>12}\n")

        f.write("\n--- Per-Regime Context-Feature Decomposition ---\n")
        for r in regimes_data:
            f.write(f"  Regime {r['index']} ({r['leader']}, windows "
                    f"{r['start']}-{r['end']}):\n")
            for c, v in r["top_channels"]:
                f.write(f"      context feature {c:>3}  {v:>12.6f}\n")
            if r["runner_up"]:
                f.write(f"    vs {r['runner_up']}:\n")
                for c, v in r["gap_channels"]:
                    f.write(f"      context feature {c:>3}  {v:>+12.6f}\n")

    print(f"Ranking explainability report saved to {output_file}")

    # ── Intermediate Representation (grounded LLM input; non-fatal) ─────────
    try:
        ir_doc = ir.build_thompson_ranking_ir(
            dataset, entity, n_windows=T,
            final_ranking=ranking,
            winner_context_features=winner_context_features,
            gap_context_features=gap_context_features,
            selection_counts=counts,
            regimes=regimes_data,
            warmup_windows=warmup_used,
            context_feature_names=payload.get("channel_names"),
            n_context_features=n_context_features,
            # Every detector's per-context-feature shares, so the page can decompose
            # ANY pair's gap without the run being present: the gap split is
            # exactly shares(a) - shares(b) (see rank_gap_decomposition), so
            # this is all the on-demand renderer needs.
            context_feature_shares={
                m: [float(v) for v in aggregate_squared_per_context_feature(mu, n_context_features)]
                for m, mu in means.items()
            } if n_context_features > 0 else {},
        )
        ir.write_stage_ir(ir_doc, dataset, entity, "ir_thompson_ranking")
    except Exception as e:
        logger.error(f"Thompson ranking IR emission failed (non-fatal): {e}")


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

        axes[i + 2].set_title(f'{algorithm} Anomaly Scores, '
                              f'F1 Score = {f1_score_value}, PR AUC = {pr_auc_value}')
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
                                 explain=False, metrics=DEFAULT_DECISION_METRICS):
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
        metrics=metrics,
        vus_win=vus_window(test_data.entities[0].Y),
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
        # Regimes are detected on the PRE-update rewards. The post-update history
        # lets the detector that was just evaluated fold that window's own reward
        # into its value before being compared against the ones that were not
        # evaluated — a self-selection bump of up to r/2 on a detector's first
        # pick, largest exactly where the early regimes form.
        regime_shifts, blip_windows = detect_regime_shifts(pre_exp_rewards_hist)
        # Written once for all three stages: the reward, SHAP and ranking frames
        # are all per-model per-context-feature vectors over the same windows, and the
        # WebUI draws whichever one is asked for rather than the pipeline
        # writing every one of them.
        save_per_window_context_features(means, shap_payload, dataset, entity, iterations)
        plot_expected_rewards(exp_rewards_hist, regime_shifts, list(trained_models.keys()),
                              dataset, entity, iterations, smooth=False)
        plot_expected_rewards(exp_rewards_hist, regime_shifts, list(trained_models.keys()),
                              dataset, entity, iterations, smooth=True)
        plot_selection_states(selection_states, dataset, entity, iterations)
        plot_shap_per_model(means, shap_payload, dataset, entity, iterations)
        # Each SHAP comparison plot is produced in both a top-k and an all-models variant.
        plot_shap_comparison(means, shap_payload, dataset, entity, iterations)
        plot_shap_comparison(means, shap_payload, dataset, entity, iterations, all_models=True)
        plot_shap_per_regime(means, shap_payload, regime_shifts, dataset, entity, iterations)
        plot_shap_per_regime(means, shap_payload, regime_shifts, dataset, entity, iterations,
                             all_models=True)
        # Kept, but demoted: mean|SHAP| measures how much a context feature's influence
        # VARIES across windows, not how much it contributes on average. The
        # run-level summary is now plot_reward_average_all.
        plot_shap_average_all(means, shap_payload, dataset, entity, iterations)
        plot_shap_average_all(means, shap_payload, dataset, entity, iterations, all_models=False)

        # ── Expected-reward contribution (mu^T x split per context feature) ─────────
        # Full parity with the SHAP sets above so the two can be read frame for
        # frame. These are the ones whose bars sum to the prediction; the SHAP
        # ones answer the narrower question of deviation from a typical window.
        plot_reward_per_regime(means, shap_payload, regime_shifts, dataset, entity,
                               iterations)
        plot_reward_per_regime(means, shap_payload, regime_shifts, dataset, entity,
                               iterations, all_models=True)
        plot_reward_average_all(means, shap_payload, dataset, entity, iterations)
        plot_reward_average_all(means, shap_payload, dataset, entity, iterations,
                                all_models=False)
        explain_thompson_sampling(means, exp_rewards_hist, l2_norm_hist, pre_exp_rewards_hist,
                                  list_of_chosen_models,
                                  regime_shifts, blip_windows, selection_states,
                                  shap_payload,
                                  dataset, entity, iterations)

        # ── Ranking criterion (||mu_k||^2) — the sibling stage ──────────────
        # Separate regimes from the ones above: leadership on the ranking score
        # rather than on expected reward, by plain run-length encoding of the
        # argmax. Segmented once here and handed to both the plots and the
        # report so a figure can never disagree with the sentence beside it.
        means_history = (shap_payload or {}).get("means_history") or []
        n_context_features_ranking = int((shap_payload or {}).get("n_channels") or 0)
        regime_segments, warmup_used = leadership_regimes(means_history)
        plot_ranking_criterion(means_history, regime_segments, warmup_used,
                               dataset, entity, iterations)
        plot_ranking_final(means, list_of_chosen_models, dataset, entity, iterations)
        plot_ranking_channels(means, n_context_features_ranking, dataset, entity, iterations)
        plot_ranking_channels(means, n_context_features_ranking, dataset, entity, iterations,
                              all_models=True)
        plot_ranking_gap(means, n_context_features_ranking, dataset, entity, iterations)
        plot_ranking_per_regime(means_history, regime_segments, n_context_features_ranking,
                                dataset, entity, iterations)
        explain_thompson_ranking(means, list_of_chosen_models, shap_payload,
                                 regime_segments, warmup_used,
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
