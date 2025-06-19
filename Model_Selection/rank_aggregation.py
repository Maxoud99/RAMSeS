#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

##########################################
# Functions for rank aggregation
##########################################

from itertools import combinations, permutations
from typing import Optional, Tuple, List

import cvxpy as cp
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from distributions import mallows_kendall as mk


##########################################
# Trimmed Rank Aggregators
##########################################


def trimmed_partial_borda(ranks: np.ndarray,
                          weights: Optional[np.ndarray] = None,
                          top_k: Optional[int] = None,
                          top_kr: Optional[int] = None,
                          aggregation_type='kemeny',
                          metric: str = 'influence',
                          n_neighbors: int = 6) -> Tuple[float, np.ndarray]:
    """Computes the trimmed borda rank

    Parameters
    ----------
    ranks: [# permutations, # items]
        Array of ranks

    weights: [# permutations,]
        Weights of each permutation. By default, weights=None.

    top_k: int
        How many items to consider for partial rank aggregation.
        By default top_k=None.

    top_kr: int
        How many permutations to use for rank aggregation.
        By default top_kr=None. If top is None, then use
        agglomerative clustering.

    aggregation_type: str
        Type of aggregation method to use while computing influence.
        We recommend 'borda' in large problems and kemeny in smaller
        problems.

    metric: str
        Metric of rank reliablity. By default metric='influence'.

    n_neighbors: int
        Number of neighbours to use for proximity based reliability
    """
    reliability = _get_reliability(ranks=ranks,
                                   metric=metric,
                                   aggregation_type=aggregation_type,
                                   top_k=top_k,
                                   n_neighbors=n_neighbors)

    if top_kr is None:
        trimmed_ranks = _get_trimmed_ranks_clustering(ranks, reliability)
    else:
        trimmed_ranks = ranks[np.argsort(-1 * reliability)[:top_kr], :]

    if weights is not None:
        trimmed_weights = weights[np.argsort(-1 * reliability)[:top_kr], :]
    else:
        trimmed_weights = None

    return partial_borda(ranks=trimmed_ranks,
                         weights=trimmed_weights,
                         top_k=top_k)


def trimmed_borda(ranks: np.ndarray,
                  weights: Optional[np.ndarray] = None,
                  top_k: Optional[int] = None,
                  top_kr: Optional[int] = None,
                  aggregation_type='kemeny',
                  metric: str = 'influence',
                  n_neighbors: int = 6) -> Tuple[float, np.ndarray]:
    """Computes the trimmed borda rank

    Parameters
    ----------
    ranks: [# permutations, # items]
        Array of ranks

    weights: [# permutations,]
        Weights of each permutation. By default, weights=None.

    top_k: int
        How many items to consider for partial rank aggregation.
        By default top_k=None.

    top_kr: int
        How many permutations to use for rank aggregation.
        By default top_kr=None. If top is None, then use
        agglomerative clustering.

    aggregation_type: str
        Type of aggregation method to use while computing influence.
        We recommend 'borda' in large problems and kemeny in smaller
        problems.

    metric: str
        Metric of rank reliablity. By default metric='influence'.

    n_neighbors: int
        Number of neighbours to use for proximity based reliability
    """
    reliability = _get_reliability(ranks=ranks,
                                   metric=metric,
                                   aggregation_type=aggregation_type,
                                   top_k=top_k,
                                   n_neighbors=n_neighbors)

    if top_kr is None:
        trimmed_ranks = _get_trimmed_ranks_clustering(ranks, reliability)
    else:
        trimmed_ranks = ranks[np.argsort(-1 * reliability)[:top_kr], :]

    if weights is not None:
        trimmed_weights = weights[np.argsort(-1 * reliability)[:top_kr], :]
    else:
        trimmed_weights = None

    return borda(ranks=trimmed_ranks, weights=trimmed_weights)


def trimmed_kemeny(ranks: np.ndarray,
                   weights: Optional[np.ndarray] = None,
                   top_k: Optional[int] = None,
                   top_kr: Optional[int] = None,
                   aggregation_type='kemeny',
                   metric: str = 'influence',
                   n_neighbors: int = 6,
                   verbose: bool = True) -> Tuple[float, np.ndarray]:
    """Computes the trimmed kemeny rank

    Parameters
    ----------
    ranks: [# permutations, # items]
        Array of ranks

    weights: [# permutations,]
        Weights of each permutation. By default, weights=None.

    top_k: int
        How many items to consider for partial rank aggregation.
        By default top_k=None.

    top_kr: int
        How many permutations to use for rank aggregation.
        By default top_kr=None. If top is None, then use
        agglomerative clustering.

    aggregation_type: int
        Type of aggregation method to use while computing influence.
        We recommend 'borda' in large problems and kemeny in smaller
        problems.

    metric: str
        Metric of rank reliablity. By default metric='influence'.

    n_neighbors: int
        Number of neighbours to use for proximity based reliability

    verbose: bool
        Controls verbosity
    """
    reliability = _get_reliability(ranks=ranks,
                                   metric=metric,
                                   aggregation_type=aggregation_type,
                                   top_k=top_k,
                                   n_neighbors=n_neighbors)

    if top_kr is None:
        trimmed_ranks = _get_trimmed_ranks_clustering(ranks, reliability)
    else:
        trimmed_ranks = ranks[np.argsort(-1 * reliability)[:top_kr], :]

    if weights is not None:
        trimmed_weights = weights[np.argsort(-1 * reliability)[:top_kr], :]
    else:
        trimmed_weights = None

    return kemeny(ranks=trimmed_ranks,
                  weights=trimmed_weights,
                  verbose=verbose)


##########################################
# Rank Aggregators
##########################################


##########################################
# Using average
##########################################
import numpy as np
from typing import Tuple, Optional


def average_rank_aggregator(*rankings: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes the average rank aggregation across multiple sets of rankings.

    Parameters
    ----------
    *rankings: variable number of np.ndarray
        Multiple arrays of rankings to be aggregated. Each array should be of shape [# permutations, # items].

    Returns
    -------
    Tuple[float, np.ndarray]
        A tuple containing the objective score and the aggregated rank.
    """
    # Concatenate all rankings
    combined_ranks = np.vstack(rankings)

    # Calculate the mean rank for each item
    mean_ranks = combined_ranks.mean(axis=0)

    # Objective score can be the standard deviation of the mean ranks
    objective = np.std(mean_ranks)

    # Rank items based on their mean rank (lower is better)
    aggregated_rank_positions = mean_ranks.argsort().argsort() + 1

    return objective, aggregated_rank_positions


# *******************************************
#  ==========================================


def enhanced_markov_chain_rank_aggregator(*rankings: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Aggregates multiple sets of rankings using the Markov Chain method.

    Parameters
    ----------
    *rankings: variable number of np.ndarray
        Multiple arrays of rankings to be aggregated.

    Returns
    -------
    Tuple[float, np.ndarray]
        A tuple containing the objective score and the aggregated rank.
    """
    # Concatenate all rankings and ensure they are integers
    combined_ranks = np.vstack(rankings).astype(int)

    n, m = combined_ranks.shape
    transition_matrix = np.zeros((m, m))

    # Constructing the transition matrix
    for rank in combined_ranks:
        # Ensuring zero-based indexing
        rank = rank - 1
        for j in range(m - 1):
            transition_matrix[rank[j], rank[j + 1]] += 1

    # Normalizing the transition matrix
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = np.divide(transition_matrix, row_sums[:, np.newaxis], out=np.zeros_like(transition_matrix),
                                  where=row_sums[:, np.newaxis] != 0)

    # Handling the case where some states are never visited
    transition_matrix += np.diag(np.where(row_sums == 0, 1, 0))

    # Finding stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary_distribution = np.abs(eigenvectors[:, np.argmax(eigenvalues)]).real
    stationary_distribution /= stationary_distribution.sum()

    # Objective can be the entropy of the stationary distribution
    epsilon = 1e-10
    objective = -np.sum(stationary_distribution * np.log(stationary_distribution + epsilon))

    return objective, stationary_distribution.argsort() + 1  # argsort to get ranking from scores


def calculate_transition_matrix_with_dynamic_weighting(rankings: List[List[str]], base_smoothing: float,
                                                       smoothing_factor: float) -> np.ndarray:
    unique_algorithms = sorted(set(algo for ranking in rankings for algo in ranking))
    algo_index_map = {algo: i for i, algo in enumerate(unique_algorithms)}
    num_algorithms = len(unique_algorithms)

    # Initialize pairwise comparison matrix
    pairwise_matrix = np.zeros((num_algorithms, num_algorithms))

    # Fill the pairwise comparison matrix
    for ranking in rankings:
        for i in range(len(ranking)):
            for j in range(i + 1, len(ranking)):
                pairwise_matrix[algo_index_map[ranking[i]], algo_index_map[ranking[j]]] += 1

    # Add variable smoothing factor based on rank distances
    for i in range(num_algorithms):
        for j in range(num_algorithms):
            if i != j:
                distance = abs(i - j)
                pairwise_matrix[i, j] += base_smoothing / (distance * smoothing_factor + 1)

    print("Pairwise Comparison Matrix with Dynamic Smoothing:")
    print(pairwise_matrix)

    # Apply dynamic weighting based on pairwise differences
    transition_matrix = np.zeros((num_algorithms, num_algorithms))
    for i in range(num_algorithms):
        for j in range(num_algorithms):
            if i != j:
                diff = pairwise_matrix[i, j] - pairwise_matrix[j, i]
                weight = 1 / (1 + np.exp(-diff))  # Sigmoid function for dynamic weighting
                transition_matrix[i, j] = weight

    return transition_matrix / transition_matrix.sum(axis=1, keepdims=True)


def enhanced_markov_chain_rank_aggregator_text(rankings: List[List[str]], base_smoothing: float = 1e-1,
                                               smoothing_factor: float = 0.5) -> Tuple[float, List[str]]:
    """
    Aggregates multiple sets of rankings using the Markov Chain method with dynamic smoothing adjustment.

    Parameters
    ----------
    rankings: List of lists of strings
        Multiple lists containing named rankings of algorithms to be aggregated.
    base_smoothing: float
        Base smoothing factor to balance the pairwise comparison influence.
    smoothing_factor: float
        Factor to adjust the base smoothing based on rank positions.

    Returns
    -------
    Tuple[float, List[str]]
        A tuple containing the objective score and the aggregated rank in terms of the algorithm names.
    """
    transition_matrix = calculate_transition_matrix_with_dynamic_weighting(rankings, base_smoothing, smoothing_factor)

    # Finding stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary_distribution = np.abs(eigenvectors[:, np.argmax(np.isclose(eigenvalues, 1))]).real
    stationary_distribution /= stationary_distribution.sum()

    # Objective can be the entropy of the stationary distribution
    epsilon = 1e-10
    objective = -np.sum(stationary_distribution * np.log(stationary_distribution + epsilon))

    # Extract all unique algorithm names and map them to indices
    unique_algorithms = sorted(set(algo for ranking in rankings for algo in ranking))

    # Convert the numeric rankings back to algorithm names for the output
    sorted_indices = np.argsort(stationary_distribution)[::-1]
    sorted_algorithms = [unique_algorithms[idx] for idx in sorted_indices]

    print("Stationary Distribution:")
    print(stationary_distribution)
    print("Sorted Indices:")
    print(sorted_indices)
    print("Sorted Algorithms:")
    print(sorted_algorithms)

    return objective, sorted_algorithms[::-1]


def enhanced_markov_chain_rank_aggregator_text_old(rankings: List[List[str]]) -> Tuple[float, List[str]]:
    """
    Aggregates multiple sets of rankings using the Markov Chain method, accepting named algorithms instead of numeric indices.

    Parameters
    ----------
    rankings: List of lists of strings
        Multiple lists containing named rankings of algorithms to be aggregated.

    Returns
    -------
    Tuple[float, List[str]]
        A tuple containing the objective score and the aggregated rank in terms of the algorithm names.
    """
    # Extract all unique algorithm names and map them to indices
    unique_algorithms = sorted(set(algo for ranking in rankings for algo in ranking))
    algo_index_map = {algo: i for i, algo in enumerate(unique_algorithms)}

    # Convert text rankings to numerical rankings using the map
    numeric_rankings = [np.array([algo_index_map[algo] for algo in ranking]) for ranking in rankings]

    # Concatenate all rankings and ensure they are integers
    combined_ranks = np.vstack(numeric_rankings).astype(int)

    n, m = combined_ranks.shape
    transition_matrix = np.zeros((m, m))

    # Constructing the transition matrix
    for rank in combined_ranks:
        for j in range(m - 1):
            transition_matrix[rank[j], rank[j + 1]] += 1

    # Normalizing the transition matrix
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = np.divide(transition_matrix, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis] != 0)

    # Handling the case where some states are never visited
    transition_matrix += np.diag(np.where(row_sums == 0, 1, 0))

    # Finding stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary_distribution = np.abs(eigenvectors[:, np.argmax(np.isclose(eigenvalues, 1))]).real
    stationary_distribution /= stationary_distribution.sum()

    # Objective can be the entropy of the stationary distribution
    epsilon = 1e-10
    objective = -np.sum(stationary_distribution * np.log(stationary_distribution + epsilon))

    # Convert the numeric rankings back to algorithm names for the output
    sorted_indices = stationary_distribution.argsort()
    sorted_algorithms = [unique_algorithms[idx] for idx in sorted_indices]

    return objective, sorted_algorithms


# *********************************************************
# #########################################################

def copeland_rank_aggregator(*rankings: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Aggregates multiple sets of rankings using the Copeland method.

    Parameters
    ----------
    *rankings: variable number of np.ndarray
        Multiple arrays of rankings to be aggregated.

    Returns
    -------
    Tuple[float, np.ndarray]
        A tuple containing the objective score and the aggregated rank.
    """
    combined_ranks = np.vstack(rankings).astype(int)
    n, m = combined_ranks.shape
    copeland_scores = np.zeros(m)

    # Pairwise comparisons to calculate Copeland scores
    for i in range(m):
        for j in range(i + 1, m):
            wins_i = np.sum(combined_ranks[:, i] < combined_ranks[:, j])
            wins_j = np.sum(combined_ranks[:, j] < combined_ranks[:, i])
            copeland_scores[i] += wins_i
            copeland_scores[j] += wins_j

    # Objective can be the variance of the Copeland scores
    objective = np.var(copeland_scores)

    # The final rank is based on Copeland scores, highest score gets the highest rank
    aggregated_rank = copeland_scores.argsort()[::-1] + 1

    return objective, aggregated_rank


# *********************************************************
# #########################################################
from scipy.optimize import linear_sum_assignment


def spearmans_footrule_aggregator(*rankings: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Aggregates multiple sets of rankings using Spearman's Footrule method.

    Parameters
    ----------
    *rankings: variable number of np.ndarray
        Multiple arrays of rankings to be aggregated.

    Returns
    -------
    Tuple[float, np.ndarray]
        A tuple containing the objective score and the aggregated rank.
    """
    combined_ranks = np.vstack(rankings).astype(int)
    n, m = combined_ranks.shape

    # Create a cost matrix for all pairwise footrule distances
    cost_matrix = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            cost_matrix[i, j] = np.sum(np.abs(combined_ranks[:, i] - j))

    # Solve the assignment problem (minimum weight matching in bipartite graphs)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # The final rank is determined by the assignment solution
    aggregated_rank = col_ind + 1

    # Objective can be the sum of distances in the optimal assignment
    objective = cost_matrix[row_ind, col_ind].sum()

    return objective, aggregated_rank


##########################################
def partial_borda(ranks: np.ndarray,
                  weights: Optional[np.ndarray] = None,
                  top_k: int = 5) -> Tuple[float, np.ndarray]:
    # Top-k Borda Rank Aggregation
    # NOTE: weights is only for compatibility, currently not using weights

    ranks = ranks.astype(float)
    ranks = np.nan_to_num(x=ranks,
                          nan=ranks.shape[1] + 1)  # If ranks already have NaNs
    # Mask higher ranks
    x, y = np.where((ranks > (top_k - 1)))
    for x_i, y_i in zip(x, y):
        ranks[x_i, y_i] = np.nan
    aggregated_rank = np.nan_to_num(x=mk.borda_partial(ranks, w=1, k=top_k),
                                    nan=ranks.shape[1] - 1).astype(int)
    objective = np.mean([mk.distance(r, aggregated_rank) for r in ranks])

    return objective, aggregated_rank


def borda(ranks: np.ndarray,
          weights: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    if weights is None:
        aggregated_rank = mk.median(ranks)
    else:
        aggregated_rank = mk.weighted_median(ranks)
    objective = np.mean([mk.distance(r, aggregated_rank) for r in ranks])
    return objective, aggregated_rank


def kemeny(ranks: np.ndarray,
           weights: Optional[np.ndarray] = None,
           verbose: bool = True) -> Tuple[float, np.ndarray]:
    """Kemeny-Young optimal rank aggregation [1]

    We include the ability to incorporate weights of metrics/permutations.

    Parameters
    ----------
    ranks:
        Permutations/Ranks
    weights:
        Weight of each rank/permutation.
    verbose:
        Controls verbosity

    References
    ----------
    [1] Conitzer, V., Davenport, A., & Kalagnanam, J. (2006, July). Improved bounds for computing Kemeny rankings. In AAAI (Vol. 6, pp. 620-626).
        https://www.aaai.org/Papers/AAAI/2006/AAAI06-099.pdf
    [2] http://vene.ro/blog/kemeny-young-optimal-rank-aggregation-in-python.html#note4\
    """

    _, n_models = ranks.shape

    # Minimize C.T * X
    edge_weights = build_graph(ranks, weights)
    C = -1 * edge_weights.ravel().reshape((-1, 1))

    # Defining variables
    X = cp.Variable((n_models ** 2, 1), boolean=True)

    # Defining the objective function
    objective = cp.Maximize(C.T @ X)

    # Defining the constraints
    idx = lambda i, j: n_models * i + j

    # Constraints for every pair
    pairwise_constraints = np.zeros(
        ((n_models * (n_models - 1)) // 2, n_models ** 2))
    for row, (i, j) in zip(pairwise_constraints,
                           combinations(range(n_models), 2)):
        row[[idx(i, j), idx(j, i)]] = 1

    # and for every cycle of length 3
    triangle_constraints = np.zeros(
        ((n_models * (n_models - 1) * (n_models - 2)), n_models ** 2))
    for row, (i, j, k) in zip(triangle_constraints,
                              permutations(range(n_models), 3)):
        row[[idx(i, j), idx(j, k), idx(k, i)]] = 1

    constraints = []
    constraints += [
        pairwise_constraints @ X == np.ones((pairwise_constraints.shape[0], 1))
    ]
    constraints += [
        triangle_constraints @ X >= np.ones((triangle_constraints.shape[0], 1))
    ]

    # Solving the problem
    problem = cp.Problem(objective, constraints)

    if verbose:
        print("Is DCP:", problem.is_dcp())
    problem.solve(verbose=verbose, warm_start=True)

    aggregated_rank = X.value.reshape((n_models, n_models)).sum(axis=1)

    objective = np.mean([mk.distance(r, aggregated_rank) for r in ranks])

    return objective, aggregated_rank


##########################################
# Helper functions
##########################################


def _get_reliability(ranks,
                     metric='influence',
                     aggregation_type='borda',
                     top_k=None,
                     n_neighbors=6):
    if metric == 'influence':
        reliability = influence(ranks,
                                aggregation_type=aggregation_type,
                                top_k=top_k)
    elif metric == 'proximity':
        reliability = proximity(ranks, n_neighbors=n_neighbors, top_k=top_k)
    elif metric == 'pagerank':
        reliability = pagerank(ranks, top_k=top_k)
    elif metric == 'averagedistance':
        reliability = averagedistance(ranks, top_k=top_k)
    return reliability


def _get_trimmed_ranks_clustering(ranks, reliability):
    clustering = AgglomerativeClustering(n_clusters=2,
                                         linkage='single').fit_predict(
        reliability.reshape((-1, 1)))

    cluster_ids, counts = np.unique(clustering, return_counts=True)
    largest_cluster_idx = cluster_ids[np.argmax(counts)]  # Largest cluster

    most_reliable_cluster_idx = np.argmax([
        np.sum(reliability[np.where(clustering == 0)[0]]),
        np.sum(reliability[np.where(clustering == 1)[0]])
    ])
    # np.sum(reliability[np.where(clustering == 2)[0]])])

    # trimmed_ranks = ranks[np.where(clustering == largest_cluster_idx)[0], :]
    trimmed_ranks = ranks[np.where(clustering == most_reliable_cluster_idx)
                          [0], :]  # <--- NOTE: We used this
    # trimmed_ranks = ranks[reliability > 0, :]

    return trimmed_ranks


def compute_weights(ranks: np.ndarray,
                    true_rank: Optional[np.ndarray] = None) -> np.ndarray:
    """Computes the weight of a data point based on its distance from the true permutation.
    """
    n_metrics, n_models = ranks.shape
    if true_rank is None: true_rank = np.arange(n_models)
    distance_from_true_rank = np.array(
        [mk.distance(perm, true_rank) for perm in ranks])
    scaler = MinMaxScaler(feature_range=(0, 9))
    weights = scaler.fit_transform(distance_from_true_rank.reshape((-1, 1)))
    weights = 1 / (1 + weights)
    return weights.reshape((-1, 1))


def build_graph(ranks: np.ndarray,
                metric_weights: Optional[np.ndarray] = None) -> np.ndarray:
    n_metrics, n_models = ranks.shape
    if metric_weights is None:
        metric_weights = np.ones((n_metrics, 1))
    else:
        metric_weights = metric_weights.reshape((-1, 1))
    edge_weights = np.zeros((n_models, n_models))

    for i, j in combinations(range(n_models), 2):
        preference = ranks[:, i] - ranks[:, j]
        h_ij = (metric_weights.T @ (preference < 0).astype(int).reshape(
            (-1, 1))).squeeze()  # prefers i to j
        h_ji = (metric_weights.T @ (preference > 0).astype(int).reshape(
            (-1, 1))).squeeze()  # prefers j to i
        if h_ij > h_ji:
            edge_weights[i, j] = h_ij - h_ji
        elif h_ij < h_ji:
            edge_weights[j, i] = h_ji - h_ij
    return edge_weights


##########################################
# Functions to compute influence
##########################################


def objective(ranks, aggregation_type='kemeny', top_k=None):
    if aggregation_type == 'borda':
        _, sigma_star = borda(ranks=ranks, weights=None)
    elif aggregation_type == 'kemeny':
        _, sigma_star = kemeny(ranks=ranks, weights=None, verbose=False)
    elif aggregation_type == 'partial_borda':
        _, sigma_star = partial_borda(ranks=ranks, weights=None, top_k=top_k)
    return np.mean([mk.distance(r, sigma_star) for r in ranks])


def influence(ranks, aggregation_type='kemeny', top_k=None) -> np.array:
    """Computes the reciprocal influence of each permutation/rank on the objective. Ranks with
    higher influence (and lower reciprocal influence) are more outlying.
    """
    N, n = ranks.shape
    objective_values = []
    tol = 1e-6

    if (aggregation_type == 'partial_borda') and (top_k is None):
        raise ValueError("top_k must be specified!")

    objective_all = objective(
        ranks, aggregation_type=aggregation_type,
        top_k=top_k)  # Objective when using all the permutations

    for i in combinations(np.arange(N), N - 1):
        objective_values.append(
            objective(ranks[i, :],
                      aggregation_type=aggregation_type,
                      top_k=top_k))

    # If removing a permutation results in a higher decrease in the objective
    # then it is more likely to be outlying
    influence = objective_all - np.array(
        objective_values[::-1])  # Reverse the list
    reliability = -influence

    # influence --
    # +ve -- metric good
    # -ve influence is bad
    # low positive influence or high positive influence?

    return reliability


def proximity(ranks, n_neighbors: int = 6, top_k=None) -> np.array:
    """Computes the proximity of each rank to its nearest neighbours. Ranks with higher proximity are more central.
    """
    if top_k is not None:
        ranks = ranks.astype(float)
        x, y = np.where((ranks > (top_k - 1)))
        for x_i, y_i in zip(x, y):
            ranks[x_i, y_i] = np.nan

    neigh = NearestNeighbors(n_neighbors=n_neighbors,
                             algorithm='ball_tree',
                             metric=mk.distance)
    neigh.fit(ranks)

    proximity = 1 / neigh.kneighbors(ranks)[0].mean(axis=1)

    return proximity


def pagerank(ranks, top_k=None) -> np.array:
    """Computes the pagerank of each rank. Higher pagerank implies that a rank is more authoritative.
    """
    if top_k is not None:
        ranks = ranks.astype(float)
        x, y = np.where((ranks > (top_k - 1)))
        for x_i, y_i in zip(x, y):
            ranks[x_i, y_i] = np.nan

    G = nx.Graph()

    # Create weighted undirected graph
    pdistmatrix = pdist(ranks, metric=mk.distance)
    m, _ = ranks.shape

    elist = []
    for i, j in combinations(np.arange(m), r=2):
        idx = m * i + j - ((i + 2) * (i + 1)) // 2
        elist.append((i, j, 1 / (1 + pdistmatrix[idx])))

    G.add_weighted_edges_from(elist)

    # Compute the pagerank of each node
    pagerank = np.array(list(nx.pagerank(G).values()))

    return pagerank


def averagedistance(ranks, top_k=None) -> np.array:
    """Computes the average distance of each rank to all other ranks.
    Lower average implies that a rank is more reliable.
    """
    if top_k is not None:
        ranks = ranks.astype(float)
        x, y = np.where((ranks > (top_k - 1)))
        for x_i, y_i in zip(x, y):
            ranks[x_i, y_i] = np.nan

    tol = 1e-6
    averagedist = squareform(pdist(ranks, metric=mk.distance)).mean(axis=1)
    return 1 / (tol + averagedist)
