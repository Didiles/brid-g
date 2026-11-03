"""Ranking, distances, and influence matrices."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize as sklearn_normalize

_SIMILARITY_INFLUENCE = frozenset({"cosine", "correlation", "spearman"})
_PROBABILITY_METRICS = frozenset({"jensenshannon", "hellinger"})


def maybe_normalize_l2(features: np.ndarray, normalize_l2: bool) -> np.ndarray:
    """Return an L2-normalized copy when ``normalize_l2`` is True."""
    if not normalize_l2:
        return features
    return sklearn_normalize(features, norm="l2")


def _as_probabilities(features: np.ndarray, metric: str) -> np.ndarray:
    """L1-normalize each row. Reject negatives and rows that sum to 0."""
    if np.any(features < 0):
        raise ValueError(f"{metric} requires non-negative features")
    totals = features.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError(
            f"{metric} requires every feature row to have a positive sum"
        )
    return features / totals


def _inverse_covariance(features: np.ndarray) -> np.ndarray:
    """Inverse of the sample covariance across rows (``ddof=1``)."""
    cov = np.atleast_2d(np.cov(features, rowvar=False, ddof=1))
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0 or not np.isfinite(logdet):
        raise ValueError(
            "mahalanobis covariance is not invertible "
            "(need more samples than features, and no constant feature)"
        )
    vi = np.linalg.inv(cov)
    if not np.isfinite(vi).all():
        raise ValueError(
            "mahalanobis covariance is not invertible "
            "(need more samples than features, and no constant feature)"
        )
    return vi


def _feature_variances(features: np.ndarray) -> np.ndarray:
    """Sample variance of each column (``ddof=1``)."""
    variances = np.atleast_1d(np.var(features, axis=0, ddof=1).astype(np.float64))
    if np.any(variances <= 0) or not np.isfinite(variances).all():
        raise ValueError(
            "seuclidean requires a positive finite variance on every feature"
        )
    return variances


def _distance_matrix(
    features: np.ndarray,
    metric: str,
    minkowski_p: float,
) -> np.ndarray:
    if metric == "braycurtis":
        if np.any(features < 0):
            raise ValueError("braycurtis requires non-negative features")
        with np.errstate(divide="ignore", invalid="ignore"):
            return pairwise_distances(features, metric="braycurtis")
    if metric in _PROBABILITY_METRICS:
        probs = _as_probabilities(features, metric)
        if metric == "jensenshannon":
            with np.errstate(divide="ignore", invalid="ignore"):
                return squareform(pdist(probs, metric="jensenshannon"))
        roots = np.sqrt(probs)
        return pairwise_distances(roots, metric="euclidean") / np.sqrt(2.0)
    if metric == "spearman":
        ranks = rankdata(features, method="average", axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return pairwise_distances(ranks, metric="correlation")

    kwargs = {}
    if metric == "minkowski":
        kwargs["p"] = minkowski_p
    elif metric == "mahalanobis":
        kwargs["VI"] = _inverse_covariance(features)
    elif metric == "seuclidean":
        kwargs["V"] = _feature_variances(features)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pairwise_distances(features, metric=metric, **kwargs)


def _undefined_distance_message(metric: str) -> str:
    if metric in ("correlation", "spearman"):
        return f"{metric} is undefined for a constant feature row"
    if metric == "braycurtis":
        return "braycurtis is undefined when a pair has zero denominator"
    return "distance or influence contains NaN"


def compute_distance_and_influence(
    features: np.ndarray,
    metric: str = "euclidean",
    minkowski_p: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(dist_matrix, influence_matrix)`` of shape ``(N, N)``.

    Dissimilarities (Lp family, ``canberra``, ``braycurtis``,
    ``mahalanobis``, ``seuclidean``, ``jensenshannon``, ``hellinger``):
    ``influence = 1 / dist``. The diagonal is forced to 0. Coincident
    points or identical distributions yield ``inf`` off the diagonal.
    ``minkowski_p`` is used only for ``minkowski``.

    ``cosine``: ``influence = 1 - dist`` (cosine similarity; diagonal 0).

    ``correlation`` and ``spearman``: the ranking distance is ``1`` minus
    the Pearson or Spearman coefficient. Influence is the coefficient
    itself (diagonal 0). A constant feature row raises ``ValueError``.
    """
    dist_matrix = _distance_matrix(features, metric, minkowski_p)
    if metric in _SIMILARITY_INFLUENCE:
        influence = 1.0 - dist_matrix
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            influence = 1.0 / dist_matrix
    np.fill_diagonal(influence, 0.0)
    if np.isnan(dist_matrix).any() or np.isnan(influence).any():
        raise ValueError(_undefined_distance_message(metric))
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
    # Self has distance 0, but duplicate rows can tie and move the query off
    # column 0. Force self last so rank lists never include the query node.
    dist = np.array(dist_matrix, copy=True)
    np.fill_diagonal(dist, np.inf)
    return np.argsort(dist, axis=1)[:, :k_eff]
