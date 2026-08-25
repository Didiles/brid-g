"""Edge-list helpers (NumPy / NetworkX)."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import networkx as nx
import numpy as np


def edges_to_numpy(
    edge_list: Sequence[Sequence[int]],
    num_nodes: int,
) -> np.ndarray:
    """Convert a list of ``[u, v]`` pairs to a ``(2, E)`` int64 array."""
    if not edge_list:
        return np.empty((2, 0), dtype=np.int64)

    arr = np.asarray(edge_list, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected edge list of shape (E, 2), got {arr.shape}")
    if arr.min() < 0 or arr.max() >= num_nodes:
        raise ValueError(
            f"Invalid node indices: min={arr.min()}, max={arr.max()}, "
            f"num_nodes={num_nodes}"
        )
    return arr.T.copy()


def edges_to_networkx(
    edge_list: Sequence[Sequence[int]],
    num_nodes: int,
) -> nx.Graph:
    """Build an undirected NetworkX graph with nodes ``0 .. num_nodes-1``."""
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edge_list)
    return g


def assemble_graph(
    edge_list: Sequence[Sequence[int]],
    num_nodes: int,
    *,
    return_networkx: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, nx.Graph]]:
    """Validate edges and return NumPy ``edge_index`` plus optional NetworkX."""
    edge_index = edges_to_numpy(edge_list, num_nodes)
    if return_networkx:
        return edge_index, edges_to_networkx(edge_list, num_nodes)
    return edge_index
