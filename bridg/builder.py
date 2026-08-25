"""Orchestrate the BRID-G pipeline from a feature matrix."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import networkx as nx
import numpy as np

from .config import BridGConfig
from .io import assemble_graph
from .ranking import (
    compute_distance_and_influence,
    knn_rankings,
    maybe_normalize_l2,
)
from .similarity import get_similarity_fn
from .stages import (
    add_mandatory,
    influence_select,
    make_edge_filter,
    triangulate,
)


def build_brid_g(
    features: np.ndarray,
    config: Optional[BridGConfig] = None,
    *,
    return_networkx: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, nx.Graph]]:
    """Build an undirected BRID-G graph from a feature matrix.

    Pipeline (canonical executed order):

    1. Optional L2 normalization **for ranking only**
    2. Pairwise distances + influence; top-``k`` rankings (self excluded)
    3. Influence / dominance selection; materialize edges if reciprocal ∧ RBO
    4. Triangulation among the first ``t`` accepted neighbors (still filtered)
    5. Mandatory top-``m`` edges **without** filtering
    6. Assemble undirected edge set

    Parameters
    ----------
    features :
        Feature matrix of shape ``(N, D)``.
    config :
        Hyperparameters. Defaults to :class:`BridGConfig` paper settings.
    return_networkx :
        If True (default), also return a :class:`networkx.Graph`.

    Returns
    -------
    edge_index :
        ``np.ndarray`` of shape ``(2, E)`` and dtype ``int64``.
    G :
        Undirected NetworkX graph (only if ``return_networkx`` is True).

    Notes
    -----
    L2 normalization, when enabled, is applied only inside this function for
    distance / ranking. Pass the original ``features`` to any downstream
    classifier.
    """
    if config is None:
        config = BridGConfig()

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(
            f"features must be a 2D array (N, D); got shape {features.shape}"
        )

    num_nodes = features.shape[0]
    if num_nodes < 2:
        raise ValueError("At least 2 samples are required.")

    ranked = maybe_normalize_l2(features, config.normalize_l2)
    dist_matrix, influence = compute_distance_and_influence(
        ranked, metric=config.distance_metric
    )
    knn = knn_rankings(dist_matrix, config.k)

    rank_lists = [knn[i].tolist() for i in range(num_nodes)]
    rank_sets = [set(lst) for lst in rank_lists]
    # Effective k used for RBO depth (may be < config.k when N is small).
    top_k = knn.shape[1]

    sim_fn = get_similarity_fn(config.similarity, rbo_p=config.rbo_p)
    edge_filter = make_edge_filter(
        rank_lists,
        rank_sets,
        top_k,
        reciprocal=config.reciprocal,
        rbo_threshold=config.rbo_threshold,
        sim_fn=sim_fn,
    )

    edge_list = []
    for query in range(num_nodes):
        neighbors = rank_lists[query]

        accepted, accepted_set, infl_edges = influence_select(
            query, neighbors, influence, edge_filter
        )
        edge_list.extend(infl_edges)

        edge_list.extend(
            triangulate(accepted, config.triangulation, edge_filter)
        )

        _, mand_edges = add_mandatory(
            query, neighbors, config.mandatory, accepted_set
        )
        edge_list.extend(mand_edges)

    return assemble_graph(edge_list, num_nodes, return_networkx=return_networkx)
