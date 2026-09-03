"""Pipeline stages: influence selection, triangulation, mandatory edges."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Set, Tuple

import numpy as np

RankLists = Sequence[Sequence[int]]
RankSets = Sequence[Set[int]]
EdgeFilter = Callable[[int, int], bool]


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


def influence_select(
    query: int,
    neighbors: Sequence[int],
    influence: np.ndarray,
    edge_filter: EdgeFilter,
) -> Tuple[List[int], Set[int], List[List[int]]]:
    """Influence / dominance selection for one query node.

    For every neighbor used as an **anchor**, candidates from the same
    top-``k`` list are kept in a local dominance set when they are not
    mutually more influential toward each other than toward the query, and
    not dominated by any already accepted member of that local set.

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

    for anchor in neighbors:
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
