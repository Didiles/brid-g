"""Anchor selection for dominance comparisons."""

import numpy as np
import pytest

from bridg import BridGConfig, build_brid_g
from bridg.ranking import compute_distance_and_influence, knn_rankings
from bridg.similarity import get_set_similarity_fn, get_similarity_fn
from bridg.stages import influence_select, select_anchors


class _Draws:
    """Deterministic stand-in for ``numpy.random.Generator.random``."""

    def __init__(self, values):
        self._values = iter(values)

    def random(self) -> float:
        return next(self._values)


def _select(neighbors, rank_lists, *, mode, probability=0.5, rng=None, top_k=3, query=0):
    if rng is None:
        rng = np.random.default_rng(0)
    return select_anchors(
        query,
        neighbors,
        rank_lists,
        top_k,
        mode=mode,
        sim_fn=get_similarity_fn("jaccard"),
        set_sim_fn=get_set_similarity_fn("jaccard"),
        probability=probability,
        rng=rng,
    )


class TestSelectAnchors:
    def test_all_and_certain_random_keep_rank_order(self):
        neighbors = [1, 2, 3, 4]
        assert _select(neighbors, [], mode="all") == neighbors
        assert _select(neighbors, [], mode="random", probability=1.0) == neighbors

    def test_random_probability_zero_keeps_nobody(self):
        assert _select([1, 2, 3, 4], [], mode="random", probability=0.0) == []

    def test_probabilistic_jaccard_drops_copies_and_keeps_disjoint(self):
        rank_lists = [
            [1, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        assert _select([1, 2, 3], rank_lists, mode="probabilistic") == [2, 3]

    def test_incremental_drops_a_copy_of_the_union_and_keeps_new_ids(self):
        rank_lists = [[] for _ in range(13)]
        rank_lists[0] = [1, 2]
        rank_lists[10] = [3, 4]
        rank_lists[11] = [1, 2, 3, 4]
        rank_lists[12] = [5, 6]
        chosen = _select([10, 11, 12], rank_lists, mode="incremental", top_k=4)
        assert chosen == [10, 12]

    def test_incremental_union_grows_only_after_a_successful_draw(self):
        rank_lists = [[] for _ in range(12)]
        rank_lists[0] = [1, 2]
        rank_lists[10] = [1, 3]
        rank_lists[11] = [1, 2, 3]

        missed = _select(
            [10, 11],
            rank_lists,
            mode="incremental",
            rng=_Draws([0.9, 0.0]),
        )
        kept = _select(
            [10, 11],
            rank_lists,
            mode="incremental",
            rng=_Draws([0.0, 0.0]),
        )
        assert missed == [11]
        assert kept == [10]

    def test_same_seed_repeats_the_random_draw(self):
        neighbors = list(range(1, 21))
        first = _select(neighbors, [], mode="random", rng=np.random.default_rng(0))
        second = _select(neighbors, [], mode="random", rng=np.random.default_rng(0))
        assert first == second
        assert first == [n for n in neighbors if n in set(first)]


class TestInfluenceUsesTheAnchorList:
    def test_empty_anchor_list_adds_no_edges(self):
        influence = np.ones((4, 4))
        accepted, accepted_set, edges = influence_select(
            0, [1, 2, 3], influence, lambda _a, _b: True, anchors=[]
        )
        assert accepted == []
        assert accepted_set == set()
        assert edges == []

    def test_explicit_full_list_matches_the_default_loop(self):
        influence = np.ones((4, 4))
        neighbors = [1, 2, 3]
        default = influence_select(0, neighbors, influence, lambda _a, _b: True)
        explicit = influence_select(
            0, neighbors, influence, lambda _a, _b: True, anchors=neighbors
        )
        assert explicit == default


class TestAnchorConfig:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="anchor_selection"):
            BridGConfig(anchor_selection="median")  # type: ignore[arg-type]

    @pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
    def test_rejects_probability_outside_unit_interval(self, probability):
        with pytest.raises(ValueError, match="anchor_probability"):
            BridGConfig(anchor_probability=probability)

    def test_accepts_probability_endpoints(self):
        assert BridGConfig(anchor_probability=0).anchor_probability == 0
        assert BridGConfig(anchor_probability=1).anchor_probability == 1


class TestAnchorPipeline:
    def test_all_matches_the_default_graph(self):
        features = np.random.RandomState(42).randn(20, 8)
        default = BridGConfig(
            k=5, mandatory=2, rbo_threshold=0.0, triangulation=2
        )
        explicit = BridGConfig(
            k=5,
            mandatory=2,
            rbo_threshold=0.0,
            triangulation=2,
            anchor_selection="all",
        )
        edges_default, _ = build_brid_g(features, config=default)
        edges_all, _ = build_brid_g(features, config=explicit)
        assert np.array_equal(edges_default, edges_all)

    def test_same_seed_rebuilds_the_same_graph(self):
        features = np.random.RandomState(42).randn(20, 8)
        cfg = BridGConfig(
            k=5,
            mandatory=2,
            rbo_threshold=0.0,
            triangulation=2,
            anchor_selection="random",
            anchor_probability=0.5,
            anchor_seed=7,
        )
        first, _ = build_brid_g(features, config=cfg)
        second, _ = build_brid_g(features, config=cfg)
        assert np.array_equal(first, second)

    def test_no_anchors_leaves_only_mandatory_edges(self):
        features = np.arange(6, dtype=np.float64).reshape(-1, 1)
        cfg = BridGConfig(
            k=3,
            mandatory=1,
            rbo_threshold=None,
            triangulation=None,
            normalize_l2=False,
            reciprocal=False,
            distance_metric="euclidean",
            anchor_selection="random",
            anchor_probability=0.0,
        )
        _, graph = build_brid_g(features, config=cfg)
        dist, _ = compute_distance_and_influence(features, "euclidean")
        knn = knn_rankings(dist, k=3)
        expected = {frozenset((i, int(knn[i, 0]))) for i in range(features.shape[0])}
        assert {frozenset(edge) for edge in graph.edges} == expected
