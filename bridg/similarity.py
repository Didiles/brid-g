"""Neighborhood similarity measures (RBO and Jaccard)."""

from __future__ import annotations

from typing import Callable, Dict, Sequence

RankList = Sequence[int]


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
    x_leftover: set = set()
    y_leftover: set = set()
    stored: set = set()
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


SIMILARITY_FUNCTIONS: Dict[str, Callable[..., float]] = {
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
