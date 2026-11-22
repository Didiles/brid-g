"""Configuration defaults for BRID-G graph construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

AnchorSelection = Literal["all", "random", "probabilistic", "incremental"]
_ANCHOR_SELECTIONS = ("all", "random", "probabilistic", "incremental")

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
    anchor_selection :
        Which top-``k`` neighbors are used as dominance anchors.
        ``"all"`` (default) uses every neighbor. ``"random"`` keeps each
        neighbor independently with probability ``anchor_probability``.
        ``"probabilistic"`` keeps neighbor ``n`` with probability
        ``1 - similarity(rank(n), rank(query))``. ``"incremental"`` uses
        that same probability against the query top-``k`` unioned with the
        top-``k`` lists of anchors already chosen for that query. Neighbors
        are considered in rank order (closest first).
    anchor_probability :
        Inclusion probability for ``anchor_selection="random"``. Ignored by
        the other modes. Must be in ``[0, 1]``.
    anchor_seed :
        Seed for the draw used by ``"random"``, ``"probabilistic"``, and
        ``"incremental"``. The same seed reproduces the same anchors.
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
    anchor_selection: AnchorSelection = "all"
    anchor_probability: float = 0.5
    anchor_seed: int = 0

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
        if self.anchor_selection not in _ANCHOR_SELECTIONS:
            allowed = ", ".join(_ANCHOR_SELECTIONS)
            raise ValueError(f"anchor_selection must be one of: {allowed}")
        probability = self.anchor_probability
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not (0.0 <= probability <= 1.0)
        ):
            raise ValueError("anchor_probability must be in [0, 1]")
        if isinstance(self.anchor_seed, bool) or not isinstance(self.anchor_seed, int):
            raise ValueError("anchor_seed must be an int")
