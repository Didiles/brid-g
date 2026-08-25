"""Ranking, distances, and influence matrices."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize as sklearn_normalize


def maybe_normalize_l2(features: np.ndarray, normalize_l2: bool) -> np.ndarray:
    """Return an L2-normalized copy when ``normalize_l2`` is True."""
    if not normalize_l2:
        return features
    return sklearn_normalize(features, norm="l2")


def compute_distance_and_influence(
    features: np.ndarray,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(dist_matrix, influence_matrix)`` of shape ``(N, N)``.

    - Euclidean: ``influence = 1 / dist`` (diagonal forced to 0).
    - Cosine: ``influence = 1 - dist`` (cosine similarity; diagonal 0).
    """
    dist_matrix = pairwise_distances(features, metric=metric)
    if metric == "cosine":
        influence = 1.0 - dist_matrix
    else:
        with np.errstate(divide="ignore"):
            influence = 1.0 / dist_matrix
    np.fill_diagonal(influence, 0.0)
    return dist_matrix, influence


def knn_rankings(dist_matrix: np.ndarray, k: int) -> np.ndarray:
    """Return nearest-neighbour indices of shape ``(N, k_eff)``, self excluded.

    ``k_eff = min(k, N - 1)``. Rank lists never include the query node.
    """
    n = dist_matrix.shape[0]
    if n < 2:
        raise ValueError("At least 2 samples are required to build rankings.")
    if k < 1:
        raise ValueError("k must be >= 1")
    k_eff = min(k, n - 1)
    # argsort places each node at column 0; take the next k_eff neighbours.
    return np.argsort(dist_matrix, axis=1)[:, 1 : k_eff + 1]
