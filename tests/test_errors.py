"""Input-error reports: exception type and message."""

import networkx as nx
import numpy as np
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


@pytest.mark.parametrize("k", [0, -1, True])
def test_invalid_k(k):
    with pytest.raises(ValueError, match="k must be"):
        BridGConfig(k=k)


@pytest.mark.parametrize("mandatory", [-1, True, 1.5])
def test_invalid_mandatory(mandatory):
    with pytest.raises(ValueError, match="mandatory"):
        BridGConfig(mandatory=mandatory)


@pytest.mark.parametrize("triangulation", [-1, True])
def test_invalid_triangulation(triangulation):
    with pytest.raises(ValueError, match="triangulation"):
        BridGConfig(triangulation=triangulation)


@pytest.mark.parametrize(
    "rbo_threshold",
    [1.5, -0.1, float("nan"), float("inf"), True],
)
def test_invalid_rbo_threshold(rbo_threshold):
    with pytest.raises(ValueError, match="rbo_threshold"):
        BridGConfig(rbo_threshold=rbo_threshold)


def test_boundaries_that_still_disable_stages():
    disabled = BridGConfig(mandatory=0, triangulation=0, rbo_threshold=0.0)
    assert disabled.mandatory == 0
    assert disabled.triangulation == 0
    assert disabled.rbo_threshold == 0.0
    closed = BridGConfig(triangulation=1, rbo_threshold=1.0)
    assert closed.triangulation == 1
    assert closed.rbo_threshold == 1.0
    assert BridGConfig(rbo_threshold=None).rbo_threshold is None


@pytest.mark.parametrize("normalize_l2", [1, 0, "yes"])
def test_invalid_normalize_l2(normalize_l2):
    with pytest.raises(ValueError, match="normalize_l2"):
        BridGConfig(normalize_l2=normalize_l2)


@pytest.mark.parametrize("reciprocal", [1, 0, "yes"])
def test_invalid_reciprocal(reciprocal):
    with pytest.raises(ValueError, match="reciprocal"):
        BridGConfig(reciprocal=reciprocal)


def test_invalid_distance_metric():
    with pytest.raises(ValueError, match="distance_metric"):
        BridGConfig(distance_metric="hamming")  # type: ignore[arg-type]


@pytest.mark.parametrize("minkowski_p", [0, -1, float("nan"), float("inf")])
def test_invalid_minkowski_p(minkowski_p):
    with pytest.raises(ValueError, match="minkowski_p"):
        BridGConfig(minkowski_p=minkowski_p)


def test_spearman_constant_row_rejected():
    features = np.array(
        [[1.0, 1.0, 1.0], [0.0, 1.0, 2.0], [1.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    cfg = BridGConfig(
        k=2,
        mandatory=None,
        rbo_threshold=None,
        triangulation=None,
        normalize_l2=False,
        distance_metric="spearman",
    )
    with pytest.raises(ValueError, match="spearman"):
        build_brid_g(features, config=cfg)


def test_mahalanobis_singular_covariance_rejected():
    features = np.array(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        dtype=np.float64,
    )
    cfg = BridGConfig(
        k=2,
        mandatory=None,
        rbo_threshold=None,
        triangulation=None,
        normalize_l2=False,
        distance_metric="mahalanobis",
    )
    with pytest.raises(ValueError, match="mahalanobis"):
        build_brid_g(features, config=cfg)


def test_correlation_constant_row_rejected():
    features = np.array(
        [[1.0, 1.0, 1.0], [0.0, 1.0, 2.0], [1.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    cfg = BridGConfig(
        k=2,
        mandatory=None,
        rbo_threshold=None,
        triangulation=None,
        normalize_l2=False,
        distance_metric="correlation",
    )
    with pytest.raises(ValueError, match="correlation"):
        build_brid_g(features, config=cfg)


def test_invalid_similarity():
    with pytest.raises(ValueError, match="similarity"):
        BridGConfig(similarity="cosine")  # type: ignore[arg-type]


@pytest.mark.parametrize("rbo_p", [0.0, 1.0, -0.1, True])
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
