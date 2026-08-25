"""Parity checks against the reference ``brid_g.graphs.create_graph_brid_n_rnn``."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from bridg import BridGConfig, build_brid_g

# Allow importing the sibling experiment package when running inside BRID-N.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

brid_g = pytest.importorskip("brid_g.graphs", reason="brid_g not available")


def _undirected_edge_set(g: nx.Graph):
    return {frozenset(e) for e in g.edges()}


@pytest.mark.parametrize(
    "params",
    [
        dict(
            k=5,
            mandatory=2,
            rbo_threshold=0.1,
            triangulation=3,
            reciprocal=True,
            distance_metric="euclidean",
            normalize_l2=False,
            similarity="rbo",
        ),
        dict(
            k=8,
            mandatory=None,
            rbo_threshold=0.0,
            triangulation=2,
            reciprocal=False,
            distance_metric="cosine",
            normalize_l2=True,
            similarity="rbo",
        ),
        dict(
            k=6,
            mandatory=3,
            rbo_threshold=0.05,
            triangulation=None,
            reciprocal=True,
            distance_metric="euclidean",
            normalize_l2=True,
            similarity="jaccard",
        ),
    ],
)
def test_parity_with_brid_g(params):
    rng = np.random.RandomState(7)
    features = rng.randn(25, 10).astype(np.float64)
    ids = np.array([str(i) for i in range(features.shape[0])])

    cfg = BridGConfig(
        k=params["k"],
        mandatory=params["mandatory"],
        rbo_threshold=params["rbo_threshold"],
        triangulation=params["triangulation"],
        reciprocal=params["reciprocal"],
        distance_metric=params["distance_metric"],
        normalize_l2=params["normalize_l2"],
        similarity=params["similarity"],
    )
    _, g_new = build_brid_g(features, config=cfg)

    feats = brid_g._maybe_normalize(features, params["normalize_l2"])
    _, g_ref = brid_g.create_graph_brid_n_rnn(
        feats,
        ids,
        top_k=params["k"],
        rbo_threshold=params["rbo_threshold"],
        triangulation_n=params["triangulation"],
        mandatory_connections=params["mandatory"],
        similarity_fn=params["similarity"],
        distance_metric=params["distance_metric"],
        reciprocal_filter=params["reciprocal"],
        device="cpu",
        return_networkx=True,
    )

    assert _undirected_edge_set(g_new) == _undirected_edge_set(g_ref)
    assert g_new.number_of_nodes() == g_ref.number_of_nodes()
