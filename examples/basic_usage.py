"""Minimal example: build a BRID-G graph from a synthetic feature matrix."""

from __future__ import annotations

import numpy as np

from bridg import BridGConfig, build_brid_g


def main() -> None:
    rng = np.random.default_rng(0)
    # N samples, D-dimensional features (e.g. CNN / ViT embeddings).
    features = rng.normal(size=(100, 64))

    # Paper / quali reference defaults: k=40, m=5, θ=0.1, t=5, L2 for ranking.
    edges, graph = build_brid_g(features)

    print(f"nodes: {graph.number_of_nodes()}")
    print(f"edges: {graph.number_of_edges()}")
    print(f"edge_index shape: {edges.shape}")

    # Custom configuration
    cfg = BridGConfig(k=20, mandatory=3, rbo_threshold=0.05, triangulation=4)
    edges2, graph2 = build_brid_g(features, config=cfg)
    print(f"custom edges: {graph2.number_of_edges()}")


if __name__ == "__main__":
    main()
