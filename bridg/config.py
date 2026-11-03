"""Configuration defaults for BRID-G graph construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

DistanceMetric = Literal[
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "cosine",
    "correlation",
    "canberra",
    "braycurtis",
    "mahalanobis",
    "seuclidean",
    "jensenshannon",
    "hellinger",
    "spearman",
]
_DISTANCE_METRICS = (
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "cosine",
    "correlation",
    "canberra",
    "braycurtis",
    "mahalanobis",
    "seuclidean",
    "jensenshannon",
    "hellinger",
    "spearman",
)


@dataclass(frozen=True)
class BridGConfig:
    """Hyperparameters for BRID-G (paper / quali reference defaults).

    Parameters
    ----------
    k :
        Neighborhood size for ranking lists (self excluded).
    mandatory :
        Number of top-ranked neighbors forced as edges after triangulation
        (no reciprocity / RBO filter). Use ``None`` or ``0`` to disable.
    rbo_threshold :
        Minimum neighborhood similarity required when materializing filtered
        edges. Use ``None`` to disable the similarity gate (reciprocity only
        if ``reciprocal`` is True).
    triangulation :
        Connect pairs among the first ``t`` accepted neighbors (insertion
        order), still subject to the edge filter. Use ``None`` or ``<= 1``
        to disable.
    distance_metric :
        Lp family — ``"euclidean"`` (p=2), ``"manhattan"`` (p=1),
        ``"chebyshev"`` (p=∞), ``"minkowski"`` (order ``minkowski_p``) —
        uses influence = 1 / distance. The same rule applies to
        ``"canberra"``, ``"braycurtis"``, ``"mahalanobis"``,
        ``"seuclidean"``, ``"jensenshannon"`` and ``"hellinger"``.
        ``"cosine"`` uses influence = 1 - cosine distance.
        ``"correlation"`` and ``"spearman"`` rank by ``1 - coefficient``
        and set influence to the Pearson or Spearman coefficient itself.
        Mahalanobis covariance and per-feature variance are estimated from
        the ranking matrix. Bray-Curtis, Jensen-Shannon and Hellinger
        require non-negative rows; the last two turn each row into a
        probability.
    minkowski_p :
        Order p for ``distance_metric="minkowski"``. Must be a finite
        number >= 1. Ignored for every other metric.
    normalize_l2 :
        If True, L2-normalize features **only for ranking / influence**.
        Downstream models should still receive the original feature matrix.
    reciprocal :
        Require mutual membership in each other's top-``k`` lists.
    similarity :
        Neighborhood similarity used with ``rbo_threshold``: ``"rbo"`` or
        ``"jaccard"``.
    rbo_p :
        Persistence parameter for truncated RBO (paper default 0.9).
    """

    k: int = 40
    mandatory: Optional[int] = 5
    rbo_threshold: Optional[float] = 0.1
    triangulation: Optional[int] = 5
    distance_metric: DistanceMetric = "euclidean"
    minkowski_p: float = 3.0
    normalize_l2: bool = True
    reciprocal: bool = True
    similarity: Literal["rbo", "jaccard"] = "rbo"
    rbo_p: float = 0.9

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.distance_metric not in _DISTANCE_METRICS:
            allowed = ", ".join(_DISTANCE_METRICS)
            raise ValueError(f"distance_metric must be one of: {allowed}")
        p = self.minkowski_p
        if (
            isinstance(p, bool)
            or not isinstance(p, (int, float))
            or not math.isfinite(p)
            or p < 1
        ):
            raise ValueError("minkowski_p must be a finite number >= 1")
        if self.similarity not in ("rbo", "jaccard"):
            raise ValueError("similarity must be 'rbo' or 'jaccard'")
        if not (0.0 < self.rbo_p < 1.0):
            raise ValueError("rbo_p must be in (0, 1)")
