"""Tests for pipeline stage order and graph properties."""

import numpy as np
import networkx as nx
import pytest

from bridg import BridGConfig, build_brid_g
from bridg.ranking import compute_distance_and_influence, knn_rankings
from bridg.similarity import compute_rbo
from bridg.stages import add_mandatory


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
