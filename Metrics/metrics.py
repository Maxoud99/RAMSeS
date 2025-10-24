#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

######################################################
# Metrics to evaluate the quality of model selection
######################################################

import random
from typing import Optional, List, Union

import numpy as np
from scipy.stats import kendalltau, norm
from sklearn.metrics import ndcg_score, average_precision_score, precision_score

from Utils.vus_utils import metricor
from vus.robustness_eval import generate_curve

np.random.seed(42)
random.seed(42)


def f1_score(predict, actual):
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


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


def get_events(y_test, outlier=1, normal=0, breaks=[]):
    events = dict()
    label_prev = normal
    event = 0  # corresponds to no event
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
            elif tim in breaks:
                # A break point was hit, end current event and start new one
                event_end = tim - 1
                events[event] = (event_start, event_end)
                event += 1
                event_start = tim

        else:
            # event_by_time_true[tim] = 0
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
        event_end = tim - 1
        events[event] = (event_start, event_end)
    return events


def get_composite_fscore_raw(pred_labels, y_test, return_prec_rec=False):
    pred_labels_binary = (pred_labels > 0.5).astype(int)
    true_events = get_events(y_test)
    tp = np.sum([pred_labels[start:end + 1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp
    rec_e = tp / (tp + fn)
    prec_t = precision_score(y_test, pred_labels_binary)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c


def get_composite_fscore_from_scores(score_t_test, thres, true_events, prec_t, return_prec_rec=False):
    pred_labels = score_t_test > thres
    tp = np.sum([pred_labels[start:end + 1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp
    rec_e = tp / (tp + fn)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def best_f1_linspace(scores, labels, n_splits, segment_adjust, f1_type='standard'):
    best_threshold = 0
    best_f1 = 0
    thresholds = np.linspace(scores.min(), scores.max(), n_splits)

    if np.sum(labels) > 0:
        for threshold in thresholds:
            predict = scores >= threshold
            if segment_adjust:
                predict = adjust_predicts(score=scores, label=(labels > 0), threshold=None, pred=predict,
                                          calc_latency=False)
            if f1_type == 'standard':
                f1, *_ = f1_score(predict, labels)
            elif f1_type == 'soft':
                f1, *_ = f1_soft_score(predict, labels)

            if f1 > best_f1:
                best_threshold = threshold
                best_f1 = f1
    else:
        best_threshold = scores.max() + 1
        best_f1 = 1

    predict = scores >= best_threshold
    if segment_adjust:
        predict = adjust_predicts(score=scores, label=(labels > 0), threshold=None, pred=predict, calc_latency=False)

    if f1_type == 'standard':
        f1, precision, recall, *_ = f1_score(predict, labels)
    elif f1_type == 'soft':
        f1, precision, recall, *_ = f1_soft_score(predict, labels)

    return f1, precision, recall, predict, labels, best_threshold


def normalize_scores(scores, interval_size):
    scores_normalized = []
    for score in scores:
        n_intervals = int(np.ceil(len(score) / interval_size))
        score_normalized = []
        for i in range(n_intervals):
            min_timestamp = i * interval_size
            max_timestamp = (i + 1) * interval_size
            std = score[:max_timestamp].std()
            score_interval = score[min_timestamp:max_timestamp] / std
            score_normalized.append(score_interval)
        score_normalized = np.hstack(score_normalized)
        scores_normalized.append(score_normalized)
    return scores_normalized


def adjusted_precision_recall_f1_auc(y_true: np.ndarray, y_scores: np.ndarray, n_splits=750):
    """Function to compute adjusted precision, recall, PR-AUC (average precision) and predictions.
    """
    from sklearn.metrics import auc

    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_splits)
    adjusted_precision = np.zeros(thresholds.shape)
    adjusted_recall = np.zeros(thresholds.shape)
    adjusted_f1 = np.zeros(thresholds.shape)

    for i, threshold in enumerate(thresholds):
        y_pred = y_scores >= threshold
        y_pred = adjust_predicts(score=y_scores, label=(y_true > 0), threshold=None, pred=y_pred, calc_latency=False)
        adjusted_f1[i], adjusted_precision[i], adjusted_recall[i], *_ = f1_score(y_pred, y_true)

    best_adjusted_f1 = np.max(adjusted_f1)
    best_threshold = thresholds[np.argmax(adjusted_f1)]
    adjusted_prauc = auc(adjusted_recall, adjusted_precision)

    adjusted_y_pred = y_scores >= best_threshold
    adjusted_y_pred = adjust_predicts(score=y_scores, label=(y_true > 0), threshold=None, pred=adjusted_y_pred,
                                      calc_latency=False)

    return adjusted_precision, adjusted_recall, best_adjusted_f1, adjusted_prauc, adjusted_y_pred


def range_based_precision_recall_f1_auc(y_true: np.ndarray, y_scores: np.ndarray, n_splits=1000, window_size=1000):
    from sklearn.metrics import auc
    import numpy as np
    """Function to compute range-based precision, recall, range-based F1, PR-AUC, and predictions."""
    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_splits)
    range_precision = np.zeros(thresholds.shape)
    range_recall = np.zeros(thresholds.shape)
    range_f1 = np.zeros(thresholds.shape)
    # print(f'= y size is: {y_scores.size}')
    for i, threshold in enumerate(thresholds):

        y_pred = y_scores >= threshold
        y_pred = adjust_predicts(score=y_scores, label=(y_true > 0), threshold=None, pred=y_pred, calc_latency=False)
        # print(f'y_pred {y_pred}')
        # Calculating range-based precision, recall, and F1
        for idx in range(len(y_pred)):
            start_idx = max(0, idx - window_size)
            end_idx = min(len(y_pred), idx + window_size + 1)

            TP = np.sum((y_pred[start_idx:end_idx] == 1) & (y_true[start_idx:end_idx] == 1))
            FP = np.sum((y_pred[start_idx:end_idx] == 1) & (y_true[start_idx:end_idx] == 0))
            FN = np.sum((y_pred[start_idx:end_idx] == 0) & (y_true[start_idx:end_idx] == 1))

            precision = TP / (TP + FP + 0.00001)
            recall = TP / (TP + FN + 0.00001)
            f1 = 2 * precision * recall / (precision + recall + 0.00001)

            range_precision[i] += precision
            range_recall[i] += recall
            range_f1[i] += f1

        range_precision[i] /= len(y_pred)
        range_recall[i] /= len(y_pred)
        range_f1[i] /= len(y_pred)

    best_range_f1 = np.max(range_f1)
    best_threshold = thresholds[np.argmax(range_f1)]
    range_prauc = auc(range_recall, range_precision)

    # Adjusting final predictions
    adjusted_y_pred = y_scores >= best_threshold
    adjusted_y_pred = adjust_predicts(score=y_scores, label=(y_true > 0), threshold=None, pred=adjusted_y_pred,
                                      calc_latency=False)

    return range_precision, range_recall, best_range_f1, range_prauc, adjusted_y_pred


def get_range_vus_roc(score, labels, slidingWindow):
    grader = metricor()
    R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(labels, score, 2 * slidingWindow)
    metrics = {'R_AUC_ROC': R_AUC_ROC, 'R_AUC_PR': R_AUC_PR, 'VUS_ROC': VUS_ROC, 'VUS_PR': VUS_PR}

    return metrics


# For centrality
def kendalltau_topk(a: np.array, b: np.array, k: int = 60):
    """Kendall's Tau correlation between the top-k elements according to a
    """
    idxs = np.argsort(-a)[:k]
    return kendalltau(a[idxs], b[idxs])


def ndcg(y_true: np.ndarray,
         y_score: np.ndarray,
         top_k: Optional[int] = None) -> float:
    # Normalized Discounted Cumulative Gain of the predicted model scores
    return ndcg_score(y_true=y_true,
                      y_score=y_score,
                      ignore_ties=True,
                      k=top_k)


def average_prauc(praucs: Union[List, np.ndarray]):
    # Average PR-AUCs of all the algorithm
    return np.mean(praucs)


def mean_reciprocal_rank(y_true: np.ndarray,
                         y_score: np.ndarray,
                         top_k: Optional[int] = None) -> float:
    """
    https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    """
    if top_k is None: top_k = 1

    sorted_model_preferences = sorted(zip(np.arange(len(y_score)), y_score),
                                      key=lambda x: x[1],
                                      reverse=True)
    sorted_model_pred_ranks = [i[0] for i in sorted_model_preferences]
    sorted_model_praucs = sorted(zip(np.arange(len(y_true)), y_true),
                                 key=lambda x: x[1],
                                 reverse=True)
    sorted_model_true_ranks = [i[0] for i in sorted_model_praucs]

    top_k_reciprocal_rank = 0
    for i in range(top_k):
        for rank, model_id in enumerate(sorted_model_true_ranks):
            if model_id == sorted_model_pred_ranks[i]:
                top_k_reciprocal_rank = top_k_reciprocal_rank + (1 /
                                                                 (rank + 1))

    return top_k_reciprocal_rank / top_k


METRICS_NAMES = [
    'PR-AUC of Top-1 Predicted Model', 'PR-AUC of Top-k Predicted Model',
    'Average PR-AUC', 'Range of PR-AUC', 'PR-AUC of Best Model',
    'PR-AUC of Top-3 Best Models', "Kendall's Tau Corr."
]


def get_metric_names():
    return METRICS_NAMES


def evaluate_model_selection(prauc: np.ndarray,
                             y_pred: np.ndarray,
                             k: int = 5) -> dict:
    """Evaluation metrics for model selection. 
    
    Parameters
    ----------
    prauc: np.ndarray (N,)
        PR-AUC of each model.
    y_pred: np.ndarray (N,)
        Predicted rank of each model.
    k: int
        Computes top-k accuracy and ndcg i.e. whether the chosen model is among Top-k according to 
        true model performance. 
    Returns
    ----------
    metrics: dict
        Dictionary of evaluation metrics
    """
    chosen_model_prauc = prauc[
        y_pred[0]]  # PR-AUC of the model with the highest preference
    chosen_top_k_prauc = np.mean(prauc[y_pred[:k]])
    mean_prauc = average_prauc(prauc)  # Mean PR-AUC of all the algorithm
    highest_prauc = np.max(prauc)  # Highest PR-AUC among all the algorithm
    highest_top_k_prauc = np.mean(prauc[np.argsort(-1 * prauc)[:k]])
    range_prauc = np.max(prauc) - np.min(prauc)

    corr_k, _ = kendalltau(x=prauc, y=y_pred)  # Kendall's Tau correlation

    metrics_values = [
        chosen_model_prauc, chosen_top_k_prauc, mean_prauc, range_prauc,
        highest_prauc, highest_top_k_prauc, corr_k
    ]

    return dict(zip(METRICS_NAMES, metrics_values))


######################################################
# Metrics to perform model selection - Prediction Errors surrogate metric
######################################################


def gaussian_likelihood(Y: np.ndarray,
                        Y_hat: np.ndarray,
                        Y_sigma: np.ndarray,
                        mask: Optional[np.ndarray] = None,
                        tol: float = 1e-6) -> float:
    # tensor to numpy
    Y = Y.numpy()
    Y_hat = Y_hat.numpy()
    Y_sigma = Y_sigma.numpy()
    mask = mask.numpy()

    if np.sum(np.isnan(Y_sigma)) > 0:
        pred_std = np.std(Y - Y_hat, axis=1, keepdims=True) + tol
    else:
        pred_std = Y_sigma + tol

    likelihood = norm.pdf((Y - Y_hat) / pred_std, loc=0, scale=1)
    likelihood = mask * likelihood
    return np.sum(likelihood) / np.sum(mask)  # Average likelihood


def mse(Y: np.ndarray,
        Y_hat: np.ndarray,
        Y_sigma: np.ndarray,
        mask: Optional[np.ndarray] = None) -> float:
    r"""
    Parameters
    ----------
    Y: np.ndarray
        Target values
    Y_hat: np.ndarray
        Predicted values
    Y_sigma: np.ndarray
        Predicted standard deviation
    mask: np.ndarray
        An array of 0s and 1s where 1 indicates which elements were masked for prediction. 
    
    .. math::

        mse = mean((Y - \hat{Y})^2)
    """
    # tensor to numpy
    Y = Y.numpy()
    Y_hat = Y_hat.numpy()
    Y_sigma = Y_sigma.numpy()
    mask = mask.numpy()

    if mask is None:
        return np.mean(np.square((Y - Y_hat)))
    else:
        return (np.sum(np.square(mask * (Y - Y_hat)))) / np.sum(mask)
        # return (np.sum(np.abs(mask * (Y - Y_hat)).numpy(), axis=0)) / np.sum(mask.numpy(), axis=0)


def mae(Y: np.ndarray,
        Y_hat: np.ndarray,
        Y_sigma: np.ndarray,
        mask: Optional[np.ndarray] = None) -> float:
    r"""    
    .. math::

        mse = mean(|Y - \hat{Y}|)
    """

    # tensor to numpy
    Y = Y.numpy()
    Y_hat = Y_hat.numpy()
    Y_sigma = Y_sigma.numpy()
    mask = mask.numpy()

    if mask is None:
        return np.mean(np.abs((Y - Y_hat)))
    else:

        return (np.sum(np.abs(mask * (Y - Y_hat)))) / np.sum(mask)
        # return (np.sum(np.abs(mask * (Y - Y_hat)).numpy(),axis=0)) / np.sum(mask.numpy(),axis=0)


def mape(Y: np.ndarray,
         Y_hat: np.ndarray,
         Y_sigma: np.ndarray,
         mask: Optional[np.ndarray] = None,
         tol: float = 1e-6) -> float:
    r"""
    .. math::

        mape = mean(|Y - \hat{Y}| / |Y|))
    """
    # Add small tolerance to ensure that division by |Y| does blow up
    # tensor to numpy
    Y = Y.numpy()
    Y_hat = Y_hat.numpy()
    Y_sigma = Y_sigma.numpy()
    mask = mask.numpy()

    Y = Y + tol
    Y_hat = Y_hat + tol
    if mask is None:
        return np.mean(np.abs(Y - Y_hat) / np.abs(Y))
    else:
        return (np.sum(mask * (np.abs(Y - Y_hat) / np.abs(Y)))) / np.sum(mask)
        # return (np.sum(np.abs(mask * (Y - Y_hat)).numpy(), axis=0)) / np.sum(mask.numpy(), axis=0)


def smape(Y: np.ndarray,
          Y_hat: np.ndarray,
          Y_sigma: np.ndarray,
          mask: Optional[np.ndarray] = None,
          tol: float = 1e-6) -> float:
    r"""
    .. math::

        smape = 2 * mean(|Y - \hat{Y}| / (|Y| + |\hat{Y}|))
    """
    # Add small tolerance to ensure that division by |Y| does blow up

    # tensor to numpy
    Y = Y.numpy()
    Y_hat = Y_hat.numpy()
    Y_sigma = Y_sigma.numpy()
    mask = mask.numpy()

    Y = Y + tol
    Y_hat = Y_hat + tol
    if mask is None:
        return 2 * np.mean(np.abs(Y - Y_hat) / (np.abs(Y) + np.abs(Y_hat)))
    else:
        return 2 * (np.sum(mask * (np.abs(Y - Y_hat) / (np.abs(Y) + np.abs(Y_hat))))) / np.sum(mask)
        # return 2 * (np.sum((mask.numpy() *(np.abs(Y - Y_hat).numpy()) / (np.abs(Y).numpy() + np.abs(Y_hat).numpy()))))/ np.sum(mask.numpy())


#############################################
# Synthetic anomaly injection
#############################################


# PR_AUC Precision-Recall Area Under Curve
def prauc(Y: np.ndarray,
          Y_scores: np.ndarray,
          segment_adjust: bool = True,
          n_splits: int = 100) -> float:
    r"""
    Compute the (adjusted) area under the precision-recall curve.
    
    Parameters
    ----------
    Y: np.ndarray
        Target values
    Y_scores: np.ndarray
        Predicted scores
    segment_adjust: bool
        Whether to compute adjusted PR-AUC. 
    n_splits: int
        Number of threshold splits to compute PR-AUC. 
    """

    if not segment_adjust:
        PR_AUC = average_precision_score(y_true=Y, probas_pred=Y_scores)
    else:
        # PR_AUC = range_based_precision_recall_f1_auc(Y,
        #                                           Y_scores,
        #                                           n_splits=n_splits)[3]
        PR_AUC = adjusted_precision_recall_f1_auc(Y,
                                                  Y_scores,
                                                  n_splits=n_splits)[3]

    return PR_AUC


###############################
# Test Sequence Precision Delay
###############################


def sequence_precision_delay(y_true, y_scores, n_splits=1000, max_delay=5):
    """
    Calculate Sequence Precision Delay for scored outputs over multiple thresholds.
    """

    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_splits)
    spd_precision = np.zeros(thresholds.shape)
    spd_delay = np.zeros(thresholds.shape)

    for i, threshold in enumerate(thresholds):
        y_pred = y_scores >= threshold
        y_pred = adjust_predicts(score=y_scores, label=(y_true > 0), threshold=None, pred=y_pred, calc_latency=False)

        delay_sum = 0
        true_positives = 0

        for idx in range(len(y_pred)):
            if y_true[idx] == 1:
                for j in range(idx, min(idx + max_delay, len(y_pred))):
                    if y_pred[j] == 1:
                        true_positives += 1
                        delay_sum += j - idx
                        break

        spd_precision[i] = true_positives / np.sum(y_true) if np.sum(y_true) > 0 else 0
        spd_delay[i] = delay_sum / true_positives if true_positives > 0 else 0

    # Calculating the best SPD precision and average delay
    best_spd_precision = np.max(spd_precision)
    best_spd_delay = np.mean(spd_delay)

    return best_spd_precision, best_spd_delay


#######################################
# Mutual information
#######################################
from sklearn.metrics import mutual_info_score


def calculate_mutual_information(Y: np.ndarray, Y_scores: np.ndarray, normalize: bool = False) -> float:
    """
    Calculate the Mutual Information between the true labels and predicted scores.

    Parameters:
    Y (np.ndarray): True labels.
    Y_scores (np.ndarray): Predicted scores or features.
    normalize (bool): If True, normalize the Mutual Information between 0 and 1.

    Returns:
    float: Mutual Information score.
    """
    mi = mutual_info_score(Y, Y_scores)
    if normalize:
        # Normalize the MI score to lie between 0 and 1
        max_mi = min(np.log(np.unique(Y).size), np.log(np.unique(Y_scores).size))
        mi /= max_mi
    return mi


########################################
# Composite Difficulty Index (CDI)
########################################
def calculate_cdi(RC, NC, NA, weights=None):
    """
    Calculate the Composite Difficulty Index (CDI) based on RC, NC, and NA metrics.

    Parameters:
    RC (float): Relative Contrast score.
    NC (float): Normalized Clusteredness of Abnormal Points score.
    NA (float): Normalized Adjacency of Normal/Abnormal Cluster score.
    weights (list of float): Weights for RC, NC, and NA in the CDI calculation.
                             If None, equal weights are assumed.

    Returns:
    float: Composite Difficulty Index score.
    """
    if weights is None:
        weights = [1 / 3, 1 / 3, 1 / 3]  # Equal weights if not provided

    # Ensure the weights sum up to 1
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # CDI calculation as a weighted sum of RC, NC, and NA
    cdi = weights[0] * RC + weights[1] * NC + weights[2] * NA
    return cdi
