"""Slow stress test: paper defaults on N=1000 synthetic embeddings."""

from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pytest

from bridg import build_brid_g

N_SAMPLES = 1000
N_FEATURES = 64


@pytest.mark.slow
def test_build_brid_g_n1000():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(N_SAMPLES, N_FEATURES))

    started = time.perf_counter()
    edges, graph = build_brid_g(features)
    elapsed = time.perf_counter() - started

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    degrees = [d for _, d in graph.degree()]
    n_isolates = sum(1 for d in degrees if d == 0)
    n_components = nx.number_connected_components(graph)
    mean_degree = float(np.mean(degrees)) if degrees else 0.0
    min_degree = min(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0

    print(
        f"N={N_SAMPLES} D={N_FEATURES} | "
        f"nodes={n_nodes} edges={n_edges} edge_index_cols={edges.shape[1]} "
        f"isolates={n_isolates} components={n_components} "
        f"degree mean/min/max={mean_degree:.2f}/{min_degree}/{max_degree} | "
        f"wall={elapsed:.2f}s"
    )

    assert n_nodes == N_SAMPLES
    assert nx.number_of_selfloops(graph) == 0
    assert edges.ndim == 2 and edges.shape[0] == 2
    assert edges.dtype == np.int64
    assert n_edges >= 1
    if edges.shape[1] > 0:
        assert int(edges.min()) >= 0
        assert int(edges.max()) < N_SAMPLES
        undirected = {frozenset((int(u), int(v))) for u, v in edges.T}
        assert len(undirected) == n_edges
