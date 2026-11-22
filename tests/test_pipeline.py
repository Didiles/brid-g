"""Tests for pipeline stage order and graph properties."""

import numpy as np
import networkx as nx
import pytest

from bridg import BridGConfig, build_brid_g
from bridg.ranking import compute_distance_and_influence, knn_rankings
from bridg.similarity import compute_rbo
from bridg.stages import add_mandatory, influence_select


@pytest.fixture()
def toy_features():
    # Small, deterministic cloud with clear nearest neighbours.
    rng = np.random.RandomState(42)
    return rng.randn(20, 8).astype(np.float64)


class TestBuildBridG:
    def test_basic_shape_and_no_self_loops(self, toy_features):
        cfg = BridGConfig(
            k=5,
            mandatory=2,
            rbo_threshold=0.0,
            triangulation=2,
            normalize_l2=True,
        )
        edges, g = build_brid_g(toy_features, config=cfg)
        assert edges.shape[0] == 2
        assert edges.dtype == np.int64
        assert g.number_of_nodes() == 20
        assert nx.number_of_selfloops(g) == 0

    def test_return_numpy_only(self, toy_features):
        cfg = BridGConfig(k=4, mandatory=1, rbo_threshold=None, triangulation=None)
        out = build_brid_g(toy_features, config=cfg, return_networkx=False)
        assert isinstance(out, np.ndarray)
        assert out.shape[0] == 2

    def test_default_config_is_deterministic_and_matches_networkx(self):
        rng = np.random.RandomState(0)
        features = rng.randn(200, 16).astype(np.float64)
        first, graph = build_brid_g(features, config=BridGConfig())
        second, _graph = build_brid_g(features, config=BridGConfig())
        assert np.array_equal(first, second)
        undirected = {frozenset((int(u), int(v))) for u, v in first.T}
        assert undirected == {frozenset(edge) for edge in graph.edges()}

    @pytest.mark.parametrize(
        "metric", ["manhattan", "chebyshev", "minkowski", "correlation"]
    )
    def test_lp_and_correlation_have_no_self_loops(self, toy_features, metric):
        cfg = BridGConfig(
            k=5,
            mandatory=1,
            rbo_threshold=0.0,
            triangulation=2,
            normalize_l2=False,
            reciprocal=False,
            distance_metric=metric,
        )
        _, graph = build_brid_g(toy_features, config=cfg)
        assert graph.number_of_nodes() == 20
        assert nx.number_of_selfloops(graph) == 0

    @pytest.mark.parametrize(
        "metric", ["canberra", "mahalanobis", "seuclidean", "spearman"]
    )
    def test_signed_metrics_have_no_self_loops(self, toy_features, metric):
        cfg = BridGConfig(
            k=5,
            mandatory=1,
            rbo_threshold=0.0,
            triangulation=2,
            normalize_l2=False,
            reciprocal=False,
            distance_metric=metric,
        )
        _, graph = build_brid_g(toy_features, config=cfg)
        assert graph.number_of_nodes() == 20
        assert nx.number_of_selfloops(graph) == 0

    @pytest.mark.parametrize(
        "metric", ["braycurtis", "jensenshannon", "hellinger"]
    )
    def test_nonnegative_metrics_have_no_self_loops(self, toy_features, metric):
        features = np.abs(toy_features)
        cfg = BridGConfig(
            k=5,
            mandatory=1,
            rbo_threshold=0.0,
            triangulation=2,
            normalize_l2=False,
            reciprocal=False,
            distance_metric=metric,
        )
        _, graph = build_brid_g(features, config=cfg)
        assert graph.number_of_nodes() == 20
        assert nx.number_of_selfloops(graph) == 0


class TestMandatoryUnfiltered:
    def test_mandatory_added_despite_failing_rbo(self):
        """Top-m edges are forced even when the similarity filter would reject them."""
        # Construct features so node 0's nearest neighbour is 1, but their
        # rank lists are almost disjoint → very low RBO.
        # Place points on a line with a gap pattern that yields distinct lists.
        n = 10
        features = np.zeros((n, 2), dtype=np.float64)
        for i in range(n):
            features[i, 0] = float(i)

        # High RBO threshold that almost nothing passes.
        cfg = BridGConfig(
            k=3,
            mandatory=1,
            rbo_threshold=0.99,
            triangulation=None,
            normalize_l2=False,
            reciprocal=False,
            distance_metric="euclidean",
        )
        edges, g = build_brid_g(features, config=cfg)

        # Node 0's nearest neighbour must be 1 (mandatory).
        assert g.has_edge(0, 1)

        # Sanity: with this threshold, RBO between 0 and 1 on k=3 lists is < 0.99
        dist, _ = compute_distance_and_influence(features, "euclidean")
        knn = knn_rankings(dist, k=3)
        score = compute_rbo(knn[0].tolist(), knn[1].tolist(), 3)
        assert score < 0.99


class TestTriangulation:
    def test_triangulation_uses_accepted_prefix(self, toy_features):
        cfg = BridGConfig(
            k=6,
            mandatory=None,
            rbo_threshold=0.0,
            triangulation=3,
            normalize_l2=False,
            reciprocal=False,
        )
        _, g = build_brid_g(toy_features, config=cfg)
        # Graph should have some edges (influence + possible triangulation).
        assert g.number_of_edges() > 0


class TestAddMandatoryHelper:
    def test_skips_already_accepted(self):
        accepted = {1, 2}
        new_ids, edges = add_mandatory(0, [1, 2, 3, 4], mandatory=3, accepted_set=accepted)
        assert new_ids == [3]
        assert edges == [[0, 3]]
        assert 3 in accepted


class TestMutualReciprocity:
    def test_reciprocal_filter_cuts_unilateral_influence_edges(self):
        """Unilateral influence edge (c in top-k(q), q not in top-k(c)) is cut iff reciprocal=True."""
        # 1D cloud: q=0 has top-2 [1, 2], while c=1 has top-2 [3, 4] (excludes 0)
        # and node 2 has top-2 [0, 1] (mutual with 0). Nodes 1 and 2 lie on
        # opposite sides of 0 so neither bilaterally dominates the other w.r.t. 0.
        features = np.array([[0.0], [3.0], [-4.0], [4.2], [4.5]], dtype=np.float64)
        dist, _ = compute_distance_and_influence(features, "euclidean")
        knn = knn_rankings(dist, k=2)
        rank_sets = [set(knn[i].tolist()) for i in range(features.shape[0])]

        q, c, mutual = 0, 1, 2
        assert c in rank_sets[q] and q not in rank_sets[c]
        assert mutual in rank_sets[q] and q in rank_sets[mutual]

        base = dict(
            k=2,
            mandatory=None,
            rbo_threshold=None,
            triangulation=None,
            normalize_l2=False,
            distance_metric="euclidean",
        )
        _, g_unilateral = build_brid_g(features, config=BridGConfig(**base, reciprocal=False))
        _, g_reciprocal = build_brid_g(features, config=BridGConfig(**base, reciprocal=True))

        assert g_unilateral.has_edge(q, c)
        assert g_unilateral.has_edge(q, mutual)
        assert not g_reciprocal.has_edge(q, c)
        assert g_reciprocal.has_edge(q, mutual)


class TestInfluenceDominance:
    def test_first_candidate_tested_against_anchor_dominance(self):
        """neighbors[0] != anchor is tested for bilateral dominance against anchor (no auto-seed)."""
        # 1D cloud: query=0 -> neighbors=[1, 2, 3] with d(0,1)=3, d(0,2)=4, d(0,3)=5.
        # Since d(1,2)=1 <= min(d(0,1), d(0,2)), anchor=2 bilaterally dominates
        # neighbors[0]=1 w.r.t. query=0, whereas survivor=3 at -5.0 is not dominated.
        features = np.array([[0.0], [3.0], [4.0], [-5.0]], dtype=np.float64)
        dist, infl = compute_distance_and_influence(features, "euclidean")
        neighbors = knn_rankings(dist, k=3)[0].tolist()

        query, first_cand, anchor, survivor = 0, 1, 2, 3
        assert neighbors == [first_cand, anchor, survivor]
        assert neighbors[0] != anchor
        assert infl[anchor, first_cand] >= infl[anchor, query]
        assert infl[first_cand, anchor] >= infl[first_cand, query]

        accepted, accepted_set, edges = influence_select(
            query, neighbors, infl, lambda _a, _b: True, anchors=[anchor]
        )
        assert first_cand not in accepted_set
        assert accepted == [survivor]
        assert edges == [[query, survivor]]

        # On the 3-node sub-cloud {0, 1, 2}, both all-anchor passes reject each other.
        neighbors_pair = knn_rankings(dist[:3, :3], k=2)[0].tolist()
        accepted_all, _, edges_all = influence_select(
            query, neighbors_pair, infl[:3, :3], lambda _a, _b: True
        )
        assert accepted_all == []
        assert edges_all == []

