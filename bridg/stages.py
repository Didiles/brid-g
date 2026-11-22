"""Pipeline stages: anchor selection, influence, triangulation, mandatory edges."""

from __future__ import annotations

from typing import Callable, List, Optional, Protocol, Sequence, Set, Tuple

import numpy as np

RankLists = Sequence[Sequence[int]]
RankSets = Sequence[Set[int]]
EdgeFilter = Callable[[int, int], bool]
ListSimFn = Callable[[Sequence[int], Sequence[int], int], float]
SetSimFn = Callable[[Sequence[int], Set[int], int], float]
_ANCHOR_MODES = ("all", "random", "probabilistic", "incremental")


class RandomDraw(Protocol):
    """Source of uniform draws in ``[0, 1)``, such as ``numpy.random.Generator``."""

    def random(self) -> float: ...


def make_edge_filter(
    rank_lists: RankLists,
    rank_sets: RankSets,
    top_k: int,
    *,
    reciprocal: bool,
    rbo_threshold: Optional[float],
    sim_fn: Callable[[Sequence[int], Sequence[int], int], float],
) -> EdgeFilter:
    """Build the reciprocity ∧ similarity gate used at materialization time."""

    def check(n_a: int, n_b: int) -> bool:
        if reciprocal and n_a not in rank_sets[n_b] and n_b not in rank_sets[n_a]:
            return False
        if rbo_threshold is not None:
            score = sim_fn(rank_lists[n_a], rank_lists[n_b], top_k)
            if score < rbo_threshold:
                return False
        return True

    return check


def _keep_with_probability(score: float, rng: RandomDraw) -> bool:
    """Draw a Bernoulli trial with probability ``clip(1 - score, 0, 1)``."""
    probability = 1.0 - score
    if probability < 0.0:
        probability = 0.0
    elif probability > 1.0:
        probability = 1.0
    return rng.random() < probability


def select_anchors(
    query: int,
    neighbors: Sequence[int],
    rank_lists: RankLists,
    top_k: int,
    *,
    mode: str,
    sim_fn: ListSimFn,
    set_sim_fn: SetSimFn,
    probability: float,
    rng: RandomDraw,
) -> List[int]:
    """Choose the dominance anchors for one query, in rank order.

    ``"all"`` returns every neighbor. ``"random"`` keeps each neighbor
    independently with ``probability``. ``"probabilistic"`` keeps neighbor
    ``n`` with probability ``1 - sim(rank(n), rank(query))``.
    ``"incremental"`` uses that probability against the query top-``k``
    unioned with the top-``k`` lists of anchors already chosen. The union
    grows only when a draw succeeds, and the query set stays in it.

    A probability of 0 never selects. A probability of 1 always selects,
    because ``rng.random()`` draws from ``[0, 1)``.
    """
    if mode not in _ANCHOR_MODES:
        allowed = ", ".join(_ANCHOR_MODES)
        raise ValueError(f"anchor_selection must be one of: {allowed}")
    if mode == "all":
        return list(neighbors)
    if mode == "random":
        return [n for n in neighbors if rng.random() < probability]

    query_list = rank_lists[query]
    if mode == "probabilistic":
        chosen: List[int] = []
        for neighbor in neighbors:
            score = sim_fn(rank_lists[neighbor], query_list, top_k)
            if _keep_with_probability(score, rng):
                chosen.append(neighbor)
        return chosen

    reference = set(query_list[:top_k])
    chosen = []
    for neighbor in neighbors:
        score = set_sim_fn(rank_lists[neighbor], reference, top_k)
        if _keep_with_probability(score, rng):
            chosen.append(neighbor)
            reference.update(rank_lists[neighbor][:top_k])
    return chosen


def influence_select(
    query: int,
    neighbors: Sequence[int],
    influence: np.ndarray,
    edge_filter: EdgeFilter,
    anchors: Optional[Sequence[int]] = None,
) -> Tuple[List[int], Set[int], List[List[int]]]:
    """Influence / dominance selection for one query node.

    For every neighbor used as an **anchor**, candidates from the same
    top-``k`` list are kept in a local dominance set when they are not
    mutually more influential toward each other than toward the query, and
    not dominated by any already accepted member of that local set.

    ``anchors`` restricts that outer loop. ``None`` uses every neighbor,
    which is the historical behavior. An empty anchor list adds no edges.
    The anchor itself is a reference, not an automatic edge. Candidates stay
    the full neighbor list.

    Edges are materialized only when ``edge_filter(query, candidate)`` passes
    (reciprocity and/or RBO). Insertion order of accepted neighbors is
    preserved for later triangulation.

    Returns
    -------
    accepted :
        Accepted neighbor ids in insertion order.
    accepted_set :
        Set view of ``accepted``.
    edges :
        Directed edge list ``[query, neighbor]`` for accepted neighbors.
    """
    accepted: List[int] = []
    accepted_set: Set[int] = set()
    edges: List[List[int]] = []
    anchor_list = neighbors if anchors is None else anchors

    for anchor in anchor_list:
        local: List[int] = []

        for candidate in neighbors:
            if candidate == anchor:
                continue

            if not local:
                local.append(candidate)
                if candidate not in accepted_set and edge_filter(query, candidate):
                    accepted.append(candidate)
                    accepted_set.add(candidate)
                    edges.append([query, candidate])
                continue

            # Skip if anchor and candidate dominate each other w.r.t. query.
            if (
                influence[anchor, candidate] >= influence[anchor, query]
                and influence[candidate, anchor] >= influence[candidate, query]
            ):
                continue

            is_dominant = True
            for prior in local:
                if (
                    influence[prior, candidate] >= influence[prior, query]
                    and influence[candidate, prior] >= influence[candidate, query]
                ):
                    is_dominant = False
                    break

            if is_dominant:
                local.append(candidate)
                if candidate not in accepted_set and edge_filter(query, candidate):
                    accepted.append(candidate)
                    accepted_set.add(candidate)
                    edges.append([query, candidate])

    return accepted, accepted_set, edges


def triangulate(
    accepted: Sequence[int],
    triangulation_n: Optional[int],
    edge_filter: EdgeFilter,
) -> List[List[int]]:
    """Connect pairs among the first ``t`` accepted neighbors (insertion order).

    Pair edges are still subject to ``edge_filter``. Disabled when ``t`` is
    ``None`` or ``<= 1``.
    """
    if triangulation_n is None or triangulation_n <= 1 or len(accepted) < 2:
        return []

    elite = list(accepted[:triangulation_n])
    edges: List[List[int]] = []
    for i in range(len(elite)):
        for j in range(i + 1, len(elite)):
            n_a, n_b = elite[i], elite[j]
            if edge_filter(n_a, n_b):
                edges.append([n_a, n_b])
    return edges


def add_mandatory(
    query: int,
    neighbors: Sequence[int],
    mandatory: Optional[int],
    accepted_set: Set[int],
) -> Tuple[List[int], List[List[int]]]:
    """Force-add the top-``m`` ranked neighbors with **no** edge filter.

    Runs after triangulation. Skips neighbors already in ``accepted_set``.
    Returns newly accepted ids (insertion order) and the corresponding edges.
    """
    if mandatory is None or mandatory <= 0:
        return [], []

    limit = min(mandatory, len(neighbors))
    new_ids: List[int] = []
    edges: List[List[int]] = []
    for i in range(limit):
        neighbor = int(neighbors[i])
        if neighbor == query:
            continue
        if neighbor not in accepted_set:
            new_ids.append(neighbor)
            accepted_set.add(neighbor)
            edges.append([query, neighbor])
    return new_ids, edges
