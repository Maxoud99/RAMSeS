#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union, Callable, List

import pandas as pd
from sklearn.neighbors import NearestNeighbors

from Metrics.metrics import mse, mae, smape, mape, prauc, gaussian_likelihood, get_range_vus_roc, best_f1_linspace, \
    adjusted_precision_recall_f1_auc, sequence_precision_delay, range_based_precision_recall_f1_auc, \
    calculate_mutual_information, calculate_cdi
from Utils.vus_utils import find_length
from distributions.mallows_kendall import distance as kendalltau_dist


######################################################
# Functions to compute ranks of algorithm given their predictions
######################################################


def rank_by_centrality(predictions: dict,
                       n_neighbors: Union[List[int], int] = [2, 4, 6],
                       metric: Callable = kendalltau_dist) -> pd.DataFrame:
    """Rank algorithm based on the centrality of their anomaly score (entity score) vectors.

    Parameters
    ----------
    predictions:dict
        Predictions dictionary returned by the `evaluate_models(...)` function.

    n_neighbours:Union[List[int], int]=4
        Number of neighbours to use when computing centrality. By default use the
        3 nearest neighbours.

    metric:Callable=kendalltau_dist
        Distance metric. By default the Kendall's Tau distance is used.
    """
    CENTRALITY = {}
    MODEL_NAMES = list(predictions.keys())
    entity_score_matrix = np.stack(
        [predictions[mn]['entity_scores'].squeeze() for mn in MODEL_NAMES],
        axis=0)
    if isinstance(n_neighbors, int):
        n_neighbors = [n_neighbors]

    neigh = NearestNeighbors(n_neighbors=np.max(n_neighbors),
                             algorithm='ball_tree',
                             metric=metric)
    neigh.fit(entity_score_matrix)

    for nn in n_neighbors:
        CENTRALITY[f'CENTRALITY_{nn}'] = dict(
            zip(
                MODEL_NAMES,
                neigh.kneighbors(entity_score_matrix,
                                 n_neighbors=nn)[0].mean(axis=1)))

    return pd.DataFrame(CENTRALITY)


def rank_by_metrics(predictions: dict, n_splits=100, sliding_window=None) -> pd.DataFrame:
    """Rank algorithm based on their observed (adjusted) PR-AUCs, Best F-1 and VUS (volume under the surface).
    
    Parameters
    ----------
    predictions:dict
        Predictions dictionary returned by the `evaluate_models(...)` function.
    """
    MODEL_NAMES = list(predictions.keys())
    METRICS = {}
    # METRICS['best_spd_delay']={}
    METRICS['PR-AUC'] = {}
    METRICS['Best F-1'] = {}
    METRICS['VUS'] = {}
    METRICS['MutualInformation'] = {}
    METRICS['CDI'] = {}
    for model_name in MODEL_NAMES:
        labels = predictions[model_name]['anomaly_labels'].squeeze()
        scores = predictions[model_name]['entity_scores'].squeeze()
        _, _, f1, auc, *_ = range_based_precision_recall_f1_auc(labels, scores, n_splits)
        # _, _, f1, auc, *_ = adjusted_precision_recall_f1_auc(labels, scores, n_splits)
        best_spd_precision, best_spd_delay = sequence_precision_delay(labels, scores)
        if sliding_window is None:
            sliding_window = find_length(predictions[model_name]['Y'].flatten())

        scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores.reshape(-1, 1)).ravel()
        evaluation_scores = get_range_vus_roc(scores, labels, sliding_window)
        print(f'entity scores for model name {model_name}')
        print(scores)
        METRICS['PR-AUC'][model_name] = auc
        METRICS['Best F-1'][model_name] = f1
        METRICS['VUS'][model_name] = evaluation_scores['VUS_ROC']
        # METRICS['best_spd_delay'][model_name]=best_spd_delay
        mi_score = calculate_mutual_information(Y=labels, Y_scores=scores, normalize=True)
        METRICS['MutualInformation'][model_name] = mi_score
        distance_matrix, normal_distances, abnormal_distances, normal_centroids, abnormal_centroids = compute_distances_and_centroids(
            scores, n_clusters=2)

        # Calculate individual metrics
        rc_score = calculate_rc(distance_matrix)
        nc_score = calculate_nc(normal_distances, abnormal_distances)
        na_score = calculate_na(normal_centroids, abnormal_centroids)

        # Calculate Composite Difficulty Index
        cdi_score = calculate_cdi(rc_score, nc_score, na_score)
        METRICS['CDI'][model_name] = cdi_score

    return pd.DataFrame(METRICS)


def rank_by_forecasting_metrics(predictions: dict) -> pd.DataFrame:
    """Rank algorithm based on their forecasting performance.

    Parameters
    ----------
    predictions:dict
        Predictions dictionary returned by the `evaluate_models(...)` function.

    """
    FORECASTING_METRICS = {}
    MODEL_NAMES = list(predictions.keys())

    for model_name in MODEL_NAMES:
        fm = {}

        fm['MAE'] = mae(Y=predictions[model_name]['Y'],
                        Y_hat=predictions[model_name]['Y_hat'],
                        Y_sigma=predictions[model_name]['Y_sigma'],
                        mask=predictions[model_name]['mask'])
        fm['MSE'] = mse(Y=predictions[model_name]['Y'],
                        Y_hat=predictions[model_name]['Y_hat'],
                        Y_sigma=predictions[model_name]['Y_sigma'],
                        mask=predictions[model_name]['mask'])
        fm['SMAPE'] = smape(Y=predictions[model_name]['Y'],
                            Y_hat=predictions[model_name]['Y_hat'],
                            Y_sigma=predictions[model_name]['Y_sigma'],
                            mask=predictions[model_name]['mask'])
        fm['MAPE'] = mape(Y=predictions[model_name]['Y'],
                          Y_hat=predictions[model_name]['Y_hat'],
                          Y_sigma=predictions[model_name]['Y_sigma'],
                          mask=predictions[model_name]['mask'])
        fm['LIKELIHOOD'] = gaussian_likelihood(
            Y=predictions[model_name]['Y'],
            Y_hat=predictions[model_name]['Y_hat'],
            Y_sigma=predictions[model_name]['Y_sigma'],
            mask=predictions[model_name]['mask'])
        FORECASTING_METRICS[model_name] = fm

    return pd.DataFrame(FORECASTING_METRICS).T


def rank_by_synthetic_anomlies(predictions,
                               n_splits=100,
                               sliding_window=None) -> pd.DataFrame:
    MODEL_NAMES = list(predictions.keys())
    ANOMALY_TYPES = list(
        set([i.split('_')[2] for i in predictions[MODEL_NAMES[0]].keys()]))
    # print(f'ANOMALY_TYPES is {ANOMALY_TYPES}')
    # print(f'MODEL_NAMES is {MODEL_NAMES}')
    evaluation_scores = {}

    for model_name in MODEL_NAMES:
        es = {}
        for anomaly_type in ANOMALY_TYPES:
            print(f'model_name is {model_name},anomaly_type is {anomaly_type}')
            labels = predictions[model_name][
                f'anomalylabels_type_{anomaly_type}'].flatten()
            scores = predictions[model_name][
                f'entityscores_type_{anomaly_type}'].flatten()

            _, _, f1, auc, *_ = adjusted_precision_recall_f1_auc(labels, scores, n_splits)

            if sliding_window is None:
                T_a = predictions[model_name][f'Ta_type_{anomaly_type}'].flatten()
                sliding_window = find_length(T_a)

            scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores.reshape(-1, 1)).ravel()
            vus_scores = get_range_vus_roc(scores, labels, sliding_window)

            es[f'SYNTHETIC_F1_{anomaly_type}'] = f1
            es[f'SYNTHETIC_PR-AUC_{anomaly_type}'] = auc
            es[f'SYNTHETIC_VUS_{anomaly_type}'] = vus_scores['VUS_ROC']
            # print(f'es is {es}')
        evaluation_scores[model_name] = es

    return pd.DataFrame(evaluation_scores).T.dropna(axis=1)


# Not used
def rank_by_praucs(predictions: dict, n_splits=100) -> pd.DataFrame:
    """Rank algorithm based on their observed PR-AUCs.
    
    Parameters
    ----------
    predictions:dict
        Predictions dictionary returned by the `evaluate_models(...)` function.
    """
    MODEL_NAMES = list(predictions.keys())
    PR_AUCS = {}
    PR_AUCS['PR-AUC'] = {}
    for model_name in MODEL_NAMES:
        PR_AUCS['PR-AUC'][model_name] = prauc(
            Y=predictions[model_name]['anomaly_labels'].squeeze(),
            Y_scores=predictions[model_name]['entity_scores'].squeeze(),
            segment_adjust=True,
            n_splits=n_splits)

    return pd.DataFrame(PR_AUCS)


# Not used
def rank_by_max_F1(predictions: dict, n_splits=100) -> pd.DataFrame:
    """Rank algorithm based on their observed best F1.
    
    Parameters
    ----------
    predictions:dict
        Predictions dictionary returned by the `evaluate_models(...)` function.
    
    """
    MODEL_NAMES = list(predictions.keys())
    F1 = {}
    F1['Best F-1'] = {}
    for model_name in MODEL_NAMES:
        f1, precision, recall, predict, _, best_threshold = best_f1_linspace(
            scores=predictions[model_name]['entity_scores'].squeeze(),
            labels=predictions[model_name]['anomaly_labels'].squeeze(),
            n_splits=n_splits,
            segment_adjust=True)
        F1['Best F-1'][model_name] = f1

    return pd.DataFrame(F1)


# Not used
def rank_by_vus(predictions: dict, sliding_window: int = None) -> pd.DataFrame:
    """Rank algorithm based on their volume under the ROC surface metric

    A recent study [1] claimed that VUS-ROC is the best metrtic in terms of separability, 
    consistency and robustness.

    NOTE: VUS-ROC for multivariate timeseries is untested. 
    
    Parameters
    ----------
    predictions: dict
        Predictions dictionary returned by the `evaluate_models(...)` function.
    sliding_window: int
        Sliding window parameter for the VUS metric
    
    References
    ----------
    [1] Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series 
        Anomaly Detection
    """
    MODEL_NAMES = list(predictions.keys())
    if sliding_window is None:
        sliding_window = find_length(predictions[MODEL_NAMES[0]]['Y'].flatten())

    VUS = {}
    VUS['VUS'] = {}
    for model_name in MODEL_NAMES:
        scores = predictions[model_name]['entity_scores'].squeeze()
        labels = predictions[model_name]['anomaly_labels'].squeeze()
        scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores.reshape(-1, 1)).ravel()
        evaluation_scores = get_range_vus_roc(scores, labels, sliding_window)

        VUS['VUS'][model_name] = evaluation_scores['VUS_ROC']

    return pd.DataFrame(VUS)


# ?????????????????//////////////

# Function calculate_rc(distance_matrix)
#     Initialize an array nearest_neighbor_distance with length equal to the number of rows in distance_matrix
#
#     For each row i in distance_matrix
#         Set every diagonal element in distance_matrix to infinity
#         Find the minimum value in row i (excluding the diagonal) and assign it to nearest_neighbor_distance[i]
#
#     Calculate mean_distance as the average of each row in distance_matrix
#
#     Calculate rc as the average of mean_distance divided by the average of nearest_neighbor_distance
#
#     Return rc
# End Function


def calculate_rc(distance_matrix):
    """
    Calculate the Relative Contrast (RC) of a dataset.

    Parameters:
    distance_matrix (np.ndarray): A distance matrix where D[i, j] is the distance between point i and j.

    Returns:
    float: Relative Contrast score.
    """
    # Distance to the nearest neighbor for each point
    nearest_neighbor_distance = np.min(distance_matrix + np.diag([np.inf] * distance_matrix.shape[0]), axis=1)

    # Mean distance for each point
    mean_distance = np.mean(distance_matrix, axis=1)

    # Relative Contrast calculation
    rc = np.mean(mean_distance) / np.mean(nearest_neighbor_distance)
    return rc


def calculate_nc(normal_distances, abnormal_distances):
    """
    Calculate the Normalized Clusteredness of Abnormal Points (NC).

    Parameters:
    normal_distances (np.ndarray): Distance matrix of normal points.
    abnormal_distances (np.ndarray): Distance matrix of abnormal points.

    Returns:
    float: NC score.
    """
    avg_normal_distance = np.mean(normal_distances)
    avg_abnormal_distance = np.mean(abnormal_distances)

    nc = avg_normal_distance / avg_abnormal_distance
    return nc


def calculate_na(normal_centroids, abnormal_centroids):
    """
    Calculate Normalized Adjacency of Normal/Abnormal Cluster (NA).

    Parameters:
    normal_centroids (np.ndarray): Centroids of normal clusters.
    abnormal_centroids (np.ndarray): Centroids of abnormal clusters.

    Returns:
    float: NA score. If NA cannot be calculated due to identical centroids, returns None or a default value.
    """
    # Calculate the minimum distance between any abnormal and normal centroid
    min_distance = np.min([np.linalg.norm(nc - ac) for nc in normal_centroids for ac in abnormal_centroids])

    # Calculate the average distance between all pairs of normal centroids, excluding identical pairs
    avg_normal_distance = np.mean(
        [np.linalg.norm(nc1 - nc2) for nc1 in normal_centroids for nc2 in normal_centroids if
         not np.array_equal(nc1, nc2)]
    )

    # Check if avg_normal_distance is zero or NaN
    if not avg_normal_distance or np.isnan(avg_normal_distance):
        # returning 0 means cancelling the importance of it at all.
        return 0

    # Calculate NA
    na = min_distance / avg_normal_distance
    return na


# =====================================================================


# Set OMP_NUM_THREADS environment variable to "1"
#
# Function compute_distances_and_centroids(entity_scores, n_clusters = 2)
#     Normalize entity_scores using MinMaxScaler
#
#     If variance of normalized entity_scores is less than a small threshold (e.g., 1e-6)
#         Print a warning about low variance and clustering effectiveness
#         Return None for all outputs
#
#     Perform KMeans clustering on normalized entity_scores with n_clusters number of clusters
#     Retrieve cluster labels from the KMeans model
#
#     If any cluster contains less than 2 data points
#         Print a warning about the small cluster size
#         Return None for all outputs
#
#     Separate normalized entity_scores into normal_scores and abnormal_scores based on cluster labels
#     Compute pairwise distances within normal_scores and abnormal_scores
#     If normal_scores or abnormal_scores contains only one point, set the respective distance to zero
#
#     Extract centroids for normal and abnormal clusters from the KMeans model
#     Compute an overall distance matrix for all normalized entity_scores
#
#     Return the overall distance matrix, distances within normal_scores, distances within abnormal_scores, normal cluster centroid, and abnormal cluster centroid
# End Function


import os
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Set the environment variable to address the memory leak warning in KMeans with MKL on Windows
os.environ["OMP_NUM_THREADS"] = "1"


def compute_distances_and_centroids(entity_scores, n_clusters: int = 2):
    """
    Compute distance matrices and centroids for normal and abnormal data points based on KMeans clustering
    for a single model's entity scores.

    This function performs KMeans clustering on normalized entity scores to identify
    normal and abnormal data clusters. It then computes pairwise distances within each cluster and identifies
    their centroids. The function includes checks to ensure that clustering results are valid for subsequent
    calculations.

    Parameters:
    entity_scores (np.ndarray): Entity scores predicted by a single model.
    n_clusters (int): Number of clusters for KMeans, ideally 2 (normal and abnormal).

    Returns:
    tuple: Tuple containing the overall distance matrix, distance matrices for normal and abnormal points,
           and centroids for normal and abnormal clusters. If clustering results are ineffective (e.g., empty or
           single-member clusters), returns None for each element in the tuple to indicate failure in computation.
    """
    # Normalize the entity scores
    scores_normalized = MinMaxScaler().fit_transform(entity_scores.reshape(-1, 1)).reshape(-1)

    # Check the variance in scores to ensure clustering is viable
    if np.var(scores_normalized) < 1e-6:
        print("Warning: Low variance in scores, clustering might not be effective.")
        return None, None, None, None, None

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(scores_normalized.reshape(-1, 1))
    labels = kmeans.labels_

    # Check the distribution of labels to ensure no cluster is too small
    if np.any(np.bincount(labels) < 2):
        print("Warning: One or more clusters have less than 2 points.")
        return None, None, None, None, None

    # Extract scores for normal and abnormal clusters
    normal_scores = scores_normalized[labels == 0]
    abnormal_scores = scores_normalized[labels == 1]

    # Compute pairwise distances for normal and abnormal scores
    normal_distances = pairwise_distances(normal_scores.reshape(-1, 1)) if len(normal_scores) > 1 else np.array([[0]])
    abnormal_distances = pairwise_distances(abnormal_scores.reshape(-1, 1)) if len(abnormal_scores) > 1 else np.array(
        [[0]])

    # Extract centroids for each cluster
    normal_centroids = kmeans.cluster_centers_[0].reshape(1, -1)
    abnormal_centroids = kmeans.cluster_centers_[1].reshape(1, -1)

    # Compute overall distance matrix for the scores
    overall_distance_matrix = pairwise_distances(scores_normalized.reshape(-1, 1))

    return overall_distance_matrix, normal_distances, abnormal_distances, normal_centroids, abnormal_centroids
