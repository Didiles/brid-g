"""Neighborhood similarity measures (RBO and Jaccard)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

RankList = Sequence[int]
SetSimFn = Callable[[RankList, set[int], int], float]


def compute_jaccard(x: RankList, y: RankList, top_k: int) -> float:
    """Set Jaccard over the first ``top_k`` entries of two rank lists."""
    sx = set(x[:top_k])
    sy = set(y[:top_k])
    union = sx | sy
    if not union:
        return 0.0
    return len(sx & sy) / len(union)


def compute_rbo(
    x: RankList,
    y: RankList,
    top_k: int,
    p: float = 0.9,
) -> float:
    """Truncated Rank-Biased Overlap (residual sum to depth ``top_k``).

    Matches the reference implementation used in the BRID-G experiments:

    ``RBO = (1 - p) * sum_{d=1}^{k} p^{d-1} * |prefix_x(d) ∩ prefix_y(d)| / d``

    with no infinite-list extrapolation beyond ``k``.
    """
    x_leftover: set[int] = set()
    y_leftover: set[int] = set()
    stored: set[int] = set()
    acum_inter = 0
    score = 0.0
    limit = min(len(x), len(y), top_k)

    for i in range(limit):
        x_elm = x[i]
        y_elm = y[i]
        if x_elm not in stored and x_elm == y_elm:
            acum_inter += 1
            stored.add(x_elm)
        else:
            if x_elm not in stored:
                if x_elm in y_leftover:
                    acum_inter += 1
                    stored.add(x_elm)
                    y_leftover.remove(x_elm)
                else:
                    x_leftover.add(x_elm)
            if y_elm not in stored:
                if y_elm in x_leftover:
                    acum_inter += 1
                    stored.add(y_elm)
                    x_leftover.remove(y_elm)
                else:
                    y_leftover.add(y_elm)

        score += (p ** i) * (acum_inter / (i + 1))

    return (1.0 - p) * score


def jaccard_against_set(candidate: RankList, reference: set[int], top_k: int) -> float:
    """Jaccard of the candidate prefix against the full reference set.

    The reference is not truncated. A growing union of anchor neighborhoods
    therefore stays visible past depth ``top_k``.
    """
    selected = set(candidate[:top_k])
    union = selected | reference
    if not union:
        return 0.0
    return len(selected & reference) / len(union)


def rbo_against_set(
    candidate: RankList,
    reference: set[int],
    top_k: int,
    p: float = 0.9,
) -> float:
    """Truncated RBO of a rank list against an unordered reference set.

    ``RBO = (1 - p) * sum_{d=1}^{k} p^{d-1} * |prefix(candidate, d) ∩ reference| / d``

    Membership is tested against the whole set, so ids that entered the
    reference after the first ``k`` positions still count when they appear
    in the candidate.
    """
    limit = min(len(candidate), top_k)
    seen: set[int] = set()
    overlap = 0
    score = 0.0
    for i in range(limit):
        item = candidate[i]
        if item not in seen:
            seen.add(item)
            if item in reference:
                overlap += 1
        score += (p ** i) * (overlap / (i + 1))
    return (1.0 - p) * score


SIMILARITY_FUNCTIONS: dict[str, Callable[..., float]] = {
    "jaccard": compute_jaccard,
    "rbo": compute_rbo,
}


def get_similarity_fn(
    name: str,
    rbo_p: float = 0.9,
) -> Callable[[RankList, RankList, int], float]:
    """Return a ``(rank_a, rank_b, top_k) -> float`` callable."""
    if name == "rbo":

        def _rbo(x: RankList, y: RankList, top_k: int) -> float:
            return compute_rbo(x, y, top_k, p=rbo_p)

        return _rbo
    if name == "jaccard":
        return compute_jaccard
    raise ValueError(f"Unknown similarity function: {name!r}")


def get_set_similarity_fn(
    name: str,
    rbo_p: float = 0.9,
) -> SetSimFn:
    """Return a ``(candidate, reference_set, top_k) -> float`` callable.

    Used when the reference is a union of rank lists rather than one list.
    """
    if name == "rbo":

        def _rbo(candidate: RankList, reference: set[int], top_k: int) -> float:
            return rbo_against_set(candidate, reference, top_k, p=rbo_p)

        return _rbo
    if name == "jaccard":
        return jaccard_against_set
    raise ValueError(f"Unknown similarity function: {name!r}")
