#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

##########################################
# Functions for rank aggregation
##########################################

import os
from itertools import combinations, permutations
from typing import Optional, Tuple, List, Dict, Any
import logging

import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

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
    # Handle empty or all-empty rankings
    non_empty_rankings = [r for r in rankings if len(r) > 0]
    if len(non_empty_rankings) == 0:
        logger.warning("All rankings are empty - returning empty aggregation")
        return (0.0, [])
    
    # If only one non-empty ranking, return it directly
    if len(non_empty_rankings) == 1:
        logger.warning(f"Only one non-empty ranking - returning it directly: {non_empty_rankings[0][:5]}")
        return (0.0, non_empty_rankings[0])
    
    # Use only non-empty rankings for aggregation
    transition_matrix = calculate_transition_matrix_with_dynamic_weighting(non_empty_rankings, base_smoothing, smoothing_factor)

    # Finding stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    
    # Check if eigenvalues close to 1 exist
    close_to_one = np.isclose(eigenvalues, 1)
    if not np.any(close_to_one):
        logger.warning("No eigenvalue close to 1 found - using first non-empty ranking as fallback")
        return (0.0, non_empty_rankings[0])
    
    stationary_distribution = np.abs(eigenvectors[:, np.argmax(close_to_one)]).real
    stationary_distribution /= stationary_distribution.sum()

    # Objective can be the entropy of the stationary distribution
    epsilon = 1e-10
    objective = -np.sum(stationary_distribution * np.log(stationary_distribution + epsilon))

    # Extract all unique algorithm names and map them to indices
    unique_algorithms = sorted(set(algo for ranking in non_empty_rankings for algo in ranking))
    
    # Handle edge case where unique_algorithms might be empty
    if len(unique_algorithms) == 0:
        logger.warning("No unique algorithms found - returning empty list")
        return (0.0, [])

    # Convert the numeric rankings back to algorithm names for the output
    sorted_indices = np.argsort(stationary_distribution)[::-1]
    sorted_algorithms = [unique_algorithms[idx] for idx in sorted_indices if idx < len(unique_algorithms)]

    print("Stationary Distribution:")
    print(stationary_distribution)
    print("Sorted Indices:")
    print(sorted_indices)
    print("Sorted Algorithms:")
    print(sorted_algorithms)

    return objective, sorted_algorithms[::-1]


# ════════════════════════════════════════════════════════════════════════════
#  Rank-aggregation explainability
#    • Leave-one-out marginal contribution
#    • Kendall τ alignment
#    • Borda alignment (default arbiter for every source)
# ════════════════════════════════════════════════════════════════════════════

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


def kendall_tau_restricted(a: List[str], b: List[str]) -> float:
    """
    Kendall's tau in [-1, 1] between two rankings, restricted to the common set
    of items. Identical ordering → +1, reverse → -1, independent → ~0.

    Items missing from either ranking are dropped from the comparison. If fewer
    than 2 items remain in common, returns 0.0 (tau undefined).
    """
    common = [x for x in a if x in b]
    if len(common) < 2:
        return 0.0
    pos_a = [a.index(x) for x in common]
    pos_b = [b.index(x) for x in common]
    tau, _ = kendalltau(pos_a, pos_b)
    if tau is None or np.isnan(tau):
        return 0.0
    return float(tau)


def leave_one_out_contributions(
    rankings: List[List[str]],
    source_names: List[str],
    full_ranking: List[str],
    aggregator=None,
) -> Dict[str, float]:
    """
    For each source i, re-aggregate without it and return the normalised Kendall
    distance (1 - tau)/2 ∈ [0, 1] between full_ranking and the LOO result.
    Higher = removing this source moves the consensus more = bigger marginal contribution.

    If removing leaves 0 sources, contribution = 0.0 for that source.
    """
    if aggregator is None:
        aggregator = enhanced_markov_chain_rank_aggregator_text
    out: Dict[str, float] = {}
    for i, name in enumerate(source_names):
        loo_input = rankings[:i] + rankings[i + 1:]
        if not loo_input:
            out[name] = 0.0
            continue
        _, loo_ranking = aggregator(loo_input)
        tau = kendall_tau_restricted(full_ranking, loo_ranking)
        out[name] = (1.0 - tau) / 2.0
    return out


def kendall_tau_alignments(
    rankings: List[List[str]],
    source_names: List[str],
    full_ranking: List[str],
) -> Dict[str, float]:
    """
    For each source, return Kendall's tau against the full ranking ∈ [-1, 1].
    Higher = source agrees more with the consensus.
    """
    return {name: kendall_tau_restricted(r, full_ranking)
            for name, r in zip(source_names, rankings)}


# ─── Kept for future use ────────────────────────────────────────────────────
# Earlier design: positional-agreement Borda — measured how well each source's
# own ranking matched the consensus's positional preferences (per-source score
# in [0, 1]). Replaced by `borda_count_resolution` below, which applies Borda
# COUNT VOTING over the LOO and Kendall rankings-of-sources to produce a single
# resolved ranking. The old implementation is retained verbatim in case the
# positional-agreement variant is useful again later.
#
# def borda_alignments(
#     rankings: List[List[str]],
#     source_names: List[str],
#     full_ranking: List[str],
# ) -> Dict[str, float]:
#     """
#     Normalised Borda alignment in [0, 1] between each source and the final
#     ranking. For each model m at source position src_pos and final position
#     full_pos (over the common item set of size n):
#         score_m = (n - src_pos) * (n - full_pos)
#     The total score is normalised by the score that would be achieved if the
#     source matched the final ranking exactly (sum of squares of (n - pos)).
#     Higher = source's positional preferences align with the final's.
#     """
#     full_positions = {m: i for i, m in enumerate(full_ranking)}
#     n = len(full_ranking)
#     if n == 0:
#         return {name: 0.0 for name in source_names}
#     max_score = float(sum((n - i) ** 2 for i in range(n)))
#     out: Dict[str, float] = {}
#     for name, r in zip(source_names, rankings):
#         s = 0.0
#         for src_pos, m in enumerate(r):
#             if m in full_positions:
#                 s += (n - src_pos) * (n - full_positions[m])
#         out[name] = s / max_score if max_score > 0 else 0.0
#     return out
# ────────────────────────────────────────────────────────────────────────────


def borda_count_resolution(
    loo_scores: Dict[str, float],
    align_scores: Dict[str, float],
) -> Dict[str, float]:
    """
    Apply Borda count voting over the two rankings-of-sources implied by LOO
    contribution and Kendall τ alignment, producing a single resolved ranking.

    Mechanics
    ---------
    Each voter is one of the two rankings of sources:
        Voter 1 : sources ordered by LOO contribution, descending
        Voter 2 : sources ordered by Kendall τ alignment, descending
    Under standard Borda voting, a source at position `r` (1-based) in a
    ranking of N sources receives `(N - r)` points from that voter. The total
    Borda count is the sum across the two voters:

        borda_count(source) = (N − loo_rank) + (N − align_rank)

    The Borda count therefore lies in [0, 2(N−1)]; sources with higher counts
    are better-ranked overall. The resulting Borda ranking of sources IS the
    resolution when LOO and Kendall disagree.

    Parameters
    ----------
    loo_scores, align_scores : Dict[str, float]
        Same source names as keys.

    Returns
    -------
    Dict[str, float]
        {source: borda_count}; higher = preferred by the joint vote.
    """
    if set(loo_scores.keys()) != set(align_scores.keys()):
        raise ValueError("loo_scores and align_scores must have identical keys.")
    n = len(loo_scores)
    loo_rank   = _ranks_from_scores(loo_scores,   descending=True)
    align_rank = _ranks_from_scores(align_scores, descending=True)
    return {name: (n - loo_rank[name]) + (n - align_rank[name])
            for name in loo_scores}


def _ranks_from_scores(scores: Dict[str, float], descending: bool = True) -> Dict[str, int]:
    """Convert a {name: score} mapping into {name: 1-based rank}. Ties get equal
    average ranks; descending=True means highest score gets rank 1."""
    names = list(scores.keys())
    vals = np.array([scores[n] for n in names], dtype=float)
    if descending:
        vals = -vals
    order = np.argsort(vals, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # Average ranks for ties
    _, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=float)
    for idx, group in enumerate(inv):
        sums[group] += ranks[idx]
    avg_ranks = sums / counts
    return {n: float(avg_ranks[inv[i]]) for i, n in enumerate(names)}


def borda_verdict_per_source(
    loo_scores: Dict[str, float],
    align_scores: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    For EVERY source, report its LOO rank, Kendall-alignment rank, and the
    Borda count + Borda rank produced by `borda_count_resolution` (Borda voting
    over the two source-rankings). The Borda rank IS the resolution: when LOO
    and Kendall disagree about a source, Borda's ranking is the consensus.

    Returns a list of dicts (one per source, in source order) with keys:
        source, loo_score, loo_rank, align_score, align_rank,
        borda_count, borda_rank, pattern, lo_align_rank_delta

    'pattern' is descriptive:
        'influential_disagreer' — LOO rank ≪ alignment rank (high LOO, low Kendall)
        'redundant_agreer'    — LOO rank ≫ alignment rank (low LOO, high Kendall)
        'consistent'          — LOO and alignment ranks agree
    """
    names = list(loo_scores.keys())
    loo_rank    = _ranks_from_scores(loo_scores,   descending=True)
    align_rank  = _ranks_from_scores(align_scores, descending=True)
    borda_count = borda_count_resolution(loo_scores, align_scores)
    borda_rank  = _ranks_from_scores(borda_count,  descending=True)

    verdicts: List[Dict[str, Any]] = []
    for name in names:
        lr = loo_rank[name]
        ar = align_rank[name]
        delta = abs(lr - ar)
        if lr < ar:
            pattern = "influential_disagreer"   # high LOO, lower alignment
        elif lr > ar:
            pattern = "redundant_agreer"      # lower LOO, higher alignment
        else:
            pattern = "consistent"

        verdicts.append({
            "source":      name,
            "loo_score":   float(loo_scores[name]),
            "loo_rank":    lr,
            "align_score": float(align_scores[name]),
            "align_rank":  ar,
            "borda_count": float(borda_count[name]),
            "borda_rank":  borda_rank[name],
            "pattern":     pattern,
            "lo_align_rank_delta": float(delta),
        })
    return verdicts


def prominent_contradictions(
    verdicts: List[Dict[str, Any]],
    rank_delta_threshold: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Subset of verdicts whose |loo_rank - align_rank| >= threshold.
    Default threshold = max(2, n // 3).
    """
    n = len(verdicts)
    if n == 0:
        return []
    threshold = rank_delta_threshold if rank_delta_threshold is not None else max(2, n // 3)
    return [v for v in verdicts if v["lo_align_rank_delta"] >= threshold]


def plot_aggregation_explainability(
    source_names: List[str],
    loo_scores: Dict[str, float],
    align_scores: Dict[str, float],
    borda_counts: Dict[str, float],
    stage_name: str,
    dataset: str,
    entity: str,
    iteration: int,
) -> None:
    """
    Grouped bar chart: x = source names; three bars per source for LOO contribution,
    Kendall τ alignment, and the Borda-count resolution. Kendall τ ∈ [-1, 1] is
    shifted to [0, 1] for visual comparison; Borda counts are normalised to [0, 1]
    by dividing by their max-possible value 2(N − 1). The report retains raw values.

    Saves to:
        myresults/robust_aggregated/{dataset}/{entity}/aggregation_explainability_{stage_name}_{iteration}.png
    """
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    n = len(source_names)
    x = np.arange(n)
    width = 0.27

    max_borda  = float(2 * (n - 1)) if n > 1 else 1.0
    loo_vals   = [loo_scores[s]                      for s in source_names]
    align_vals = [(align_scores[s] + 1.0) / 2.0      for s in source_names]
    borda_vals = [borda_counts[s] / max_borda        for s in source_names]

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * n + 4), 5))
    ax.bar(x - width, loo_vals,   width, label="LOO contribution",          color="#d62728")
    ax.bar(x,         align_vals, width, label="Kendall τ (rescaled)",      color="#1f77b4")
    ax.bar(x + width, borda_vals, width, label="Borda count (normalised)",  color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(source_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (rescaled to [0, 1])")
    ax.set_title(f"Rank Aggregation Explainability — {stage_name} stage")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", frameon=False,
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

    plt.tight_layout(pad=1.2)
    directory = f"myresults/robust_aggregated/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/aggregation_explainability_{stage_name}_{iteration}.png",
                format="png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_kendall_only_alignment(
    source_names: List[str],
    align_scores: Dict[str, float],
    winner: str,
    stage_name: str,
    dataset: str,
    entity: str,
    iteration: int,
) -> None:
    """
    Bar chart of Kendall's τ between each of the two sources and the final
    ranking. The more-aligned source (the winner) is highlighted in green.

    Saves to:
        myresults/robust_aggregated/{dataset}/{entity}/aggregation_explainability_{stage_name}_kendall_only_{iteration}.png
    """
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    x = np.arange(len(source_names))
    vals = [align_scores[s] for s in source_names]
    colours = ["#2ca02c" if s == winner else "#1f77b4" for s in source_names]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(x, vals, width=0.5, color=colours)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:+.3f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(source_names, rotation=15, ha="right")
    ax.set_ylabel("Kendall's tau with final ranking")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"Kendall-tau-Only Alignment — {stage_name} stage")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout(pad=1.2)
    directory = f"myresults/robust_aggregated/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(
        f"{directory}/aggregation_explainability_{stage_name}_kendall_only_{iteration}.png",
        format="png", dpi=300, bbox_inches="tight",
    )
    plt.close()


def explain_rank_aggregation_kendall_only(
    rankings: List[List[str]],
    source_names: List[str],
    full_ranking: List[str],
    stage_name: str,
    dataset: str,
    entity: str,
    iteration: int,
) -> Optional[Dict[str, Any]]:
    """
    Kendall-tau-only explainability for a TWO-source aggregation.

    When exactly two ranking lists feed an aggregation (as in the final stage,
    Robust_Aggregated + Thompson_Sampling), leave-one-out and Borda voting become
    degenerate — removing one source simply leaves the other untouched. A direct
    Kendall's tau comparison is the cleaner diagnostic: whichever source has the
    higher tau with the final ranking is the one the consensus leans toward.

    This is an alternative to (not a replacement for) explain_rank_aggregation —
    both sets of outputs are produced so the two methods can be compared.

    Produces (only when len(rankings) == 2):
        aggregation_explainability_{stage_name}_kendall_only_{iteration}.txt
        aggregation_explainability_{stage_name}_kendall_only_{iteration}.png

    Returns None when the source count is not 2; otherwise a dict with the tau
    values and the verdict.
    """
    if len(rankings) != 2 or len(source_names) != 2:
        return None

    align = kendall_tau_alignments(rankings, source_names, full_ranking)
    ranked = sorted(align.items(), key=lambda x: x[1], reverse=True)
    winner, winner_tau = ranked[0]
    runner, runner_tau = ranked[1]
    gap = winner_tau - runner_tau

    plot_kendall_only_alignment(source_names, align, winner,
                                stage_name, dataset, entity, iteration)

    directory = f"myresults/robust_aggregated/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    report_path = os.path.join(
        directory,
        f"aggregation_explainability_{stage_name}_kendall_only_{iteration}.txt",
    )
    with open(report_path, "w") as f:
        f.write(f"=== Rank Aggregation Explainability (Kendall-tau-only method) "
                f"— {stage_name} stage ===\n")
        f.write(f"Dataset: {dataset}  |  Entity: {entity}  |  Iteration: {iteration}\n")
        f.write(f"Sources (n=2): {', '.join(source_names)}\n")
        f.write(f"Final ranking: {full_ranking}\n\n")
        f.write("This method applies only when exactly two ranking lists feed the\n")
        f.write("aggregation. With two sources, leave-one-out and Borda voting are\n")
        f.write("degenerate (removing one source leaves the other unchanged), so a\n")
        f.write("direct Kendall's tau comparison is the cleaner diagnostic.\n\n")

        f.write("--- Kendall's tau Alignment with the Final Ranking ---\n")
        f.write(f"  {'Source':<22} {'Kendall tau':>12}\n")
        f.write("  " + "-" * 35 + "\n")
        for name in source_names:
            f.write(f"  {name:<22} {align[name]:>+12.4f}\n")

        f.write("\n--- Verdict ---\n")
        f.write(f"The final ranking is most aligned with: {winner} "
                f"(tau = {winner_tau:+.4f}).\n")
        f.write(f"{runner} is less aligned (tau = {runner_tau:+.4f}).\n")
        f.write(f"Alignment gap: {gap:.4f}.\n")

    return {
        "align_scores": align,
        "winner": winner,
        "winner_tau": winner_tau,
        "runner_up": runner,
        "runner_up_tau": runner_tau,
        "alignment_gap": gap,
    }


def explain_rank_aggregation(
    rankings: List[List[str]],
    source_names: List[str],
    full_ranking: List[str],
    stage_name: str,
    dataset: str,
    entity: str,
    iteration: int,
    explain: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Compute LOO contributions, Kendall τ alignments, Borda alignments, then for
    every source decide via Borda whether the LOO or Kendall perspective wins.
    Write a structured text report and a grouped bar plot to:
        myresults/robust_aggregated/{dataset}/{entity}/
    Files:
        aggregation_explainability_{stage_name}_{iteration}.txt
        aggregation_explainability_{stage_name}_{iteration}.png

    When exactly two ranking lists feed the aggregation (the final stage), an
    additional Kendall-τ-only analysis is produced as well — see
    explain_rank_aggregation_kendall_only — so both methods are visible side by
    side. The two extra files carry a _kendall_only suffix.

    Returns the dict of computed scores + verdicts when explain=True; None otherwise.
    """
    if not explain:
        return None
    if len(rankings) != len(source_names):
        raise ValueError(
            f"rankings ({len(rankings)}) and source_names ({len(source_names)}) "
            "must have the same length."
        )

    loo = leave_one_out_contributions(rankings, source_names, full_ranking)
    align = kendall_tau_alignments(rankings, source_names, full_ranking)
    borda_counts = borda_count_resolution(loo, align)
    verdicts = borda_verdict_per_source(loo, align)
    prominent = prominent_contradictions(verdicts)

    plot_aggregation_explainability(
        source_names, loo, align, borda_counts,
        stage_name, dataset, entity, iteration,
    )

    directory = f"myresults/robust_aggregated/{dataset}/{entity}/"
    os.makedirs(directory, exist_ok=True)
    report_path = os.path.join(
        directory, f"aggregation_explainability_{stage_name}_{iteration}.txt"
    )
    with open(report_path, "w") as f:
        f.write(f"=== Rank Aggregation Explainability — {stage_name} stage ===\n")
        f.write(f"Dataset: {dataset}  |  Entity: {entity}  |  Iteration: {iteration}\n")
        f.write(f"Sources (n={len(source_names)}): {', '.join(source_names)}\n")
        f.write(f"Final ranking: {full_ranking}\n\n")

        f.write("--- Per-Source Scores ---\n")
        f.write(f"{'Source':<22} {'LOO contrib.':>12} {'Kendall τ':>12} {'Borda count':>12}\n")
        f.write("-" * 62 + "\n")
        for name in source_names:
            f.write(
                f"{name:<22} {loo[name]:>12.4f} {align[name]:>+12.4f} "
                f"{borda_counts[name]:>12.2f}\n"
            )

        f.write("\n--- Per-Source Ranks (1 = best by that criterion;"
                " Borda rank IS the resolved ranking) ---\n")
        f.write(f"{'Source':<22} {'LOO':>5} {'Align':>6} {'Borda':>6}  {'Pattern'}\n")
        f.write("-" * 70 + "\n")
        for v in verdicts:
            f.write(
                f"{v['source']:<22} {v['loo_rank']:>5.1f} {v['align_rank']:>6.1f} "
                f"{v['borda_rank']:>6.1f}  {v['pattern']}\n"
            )

        # Show the final Borda-voted ranking of sources explicitly.
        borda_sorted = sorted(verdicts, key=lambda v: v["borda_rank"])
        f.write("\n--- Borda-Resolved Source Ranking (Borda count voting"
                " over LOO and Kendall) ---\n")
        for i, v in enumerate(borda_sorted, 1):
            f.write(
                f"  {i}. {v['source']:<22} "
                f"borda_count = {v['borda_count']:.2f}   (LOO rank {v['loo_rank']:.1f}, "
                f"Align rank {v['align_rank']:.1f})\n"
            )

        f.write("\n--- Prominent Contradictions (largest LOO vs Alignment rank gaps) ---\n")
        if not prominent:
            f.write("None detected.\n")
        else:
            for v in prominent:
                f.write(
                    f"{v['source']}: LOO rank {v['loo_rank']:.1f}, "
                    f"Alignment rank {v['align_rank']:.1f} "
                    f"(delta={v['lo_align_rank_delta']:.1f}) → {v['pattern']}\n"
                    f"   Borda resolution → rank {v['borda_rank']:.1f}"
                    f" (count {v['borda_count']:.2f})\n"
                )

    # For a two-source aggregation, also run the Kendall-τ-only method so both
    # diagnostics are visible side by side.
    kendall_only = None
    if len(rankings) == 2:
        kendall_only = explain_rank_aggregation_kendall_only(
            rankings, source_names, full_ranking, stage_name, dataset, entity, iteration)

    result = {
        "loo_scores": loo,
        "align_scores": align,
        "borda_counts": borda_counts,
        "verdicts": verdicts,
        "prominent_contradictions": prominent,
        "kendall_only": kendall_only,
    }

    # ── Intermediate Representation (grounded LLM input; non-fatal) ─────────
    try:
        _ir = _ir_module()
        source_top_picks = {
            name: (rankings[i][0] if rankings[i] else "not_available")
            for i, name in enumerate(source_names)
        }
        ir_doc = _ir.build_rank_aggregation_ir(
            dataset, entity, stage_name, iteration, result,
            source_names, source_top_picks, full_ranking)
        _ir.write_stage_ir(ir_doc, dataset, entity,
                           f"ir_rank_aggregation_{stage_name}_{iteration}")
    except Exception as e:
        logger.error(f"Rank-aggregation IR emission failed (non-fatal): {e}")

    return result


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
