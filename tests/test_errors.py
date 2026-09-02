"""Input-error reports: exception type and message."""

import numpy as np
import networkx as nx
import pytest

from bridg import BridGConfig, build_brid_g


@pytest.mark.parametrize(
    "features, match",
    [
        (np.array([1.0, 2.0]), r"2D"),
        (np.zeros((2, 4, 4)), r"2D"),
        (np.zeros((0, 8)), r"At least 2"),
        (np.zeros((1, 8)), r"At least 2"),
    ],
)
def test_invalid_feature_shape(features, match):
    with pytest.raises(ValueError, match=match):
        build_brid_g(features)


def test_nan_rejected():
    features = np.zeros((3, 4), dtype=np.float64)
    features[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_brid_g(features)


def test_inf_rejected():
    features = np.zeros((3, 4), dtype=np.float64)
    features[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        build_brid_g(features)


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k(k):
    with pytest.raises(ValueError, match="k must be"):
        BridGConfig(k=k)


def test_invalid_distance_metric():
    with pytest.raises(ValueError, match="distance_metric"):
        BridGConfig(distance_metric="manhattan")  # type: ignore[arg-type]


def test_invalid_similarity():
    with pytest.raises(ValueError, match="similarity"):
        BridGConfig(similarity="cosine")  # type: ignore[arg-type]


@pytest.mark.parametrize("rbo_p", [0.0, 1.0, -0.1])
def test_invalid_rbo_p(rbo_p):
    with pytest.raises(ValueError, match="rbo_p"):
        BridGConfig(rbo_p=rbo_p)


def test_duplicate_rows_do_not_crash():
    features = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    _, graph = build_brid_g(features)
    assert graph.number_of_nodes() == 3
    assert nx.number_of_selfloops(graph) == 0


def test_k_larger_than_n_minus_one():
    rng = np.random.RandomState(0)
    features = rng.randn(5, 8).astype(np.float64)
    cfg = BridGConfig(k=100)
    _, graph = build_brid_g(features, config=cfg)
    assert graph.number_of_nodes() == 5
    assert nx.number_of_selfloops(graph) == 0
