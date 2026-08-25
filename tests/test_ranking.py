"""Tests for ranking and self-exclusion."""

import numpy as np
import pytest

from bridg.ranking import (
    compute_distance_and_influence,
    knn_rankings,
    maybe_normalize_l2,
)


@pytest.fixture()
def sample_features():
    rng = np.random.RandomState(0)
    return rng.randn(12, 5).astype(np.float64)


class TestNormalize:
    def test_maybe_normalize_false(self, sample_features):
        out = maybe_normalize_l2(sample_features, False)
        assert out is sample_features or np.allclose(out, sample_features)

    def test_maybe_normalize_true(self, sample_features):
        out = maybe_normalize_l2(sample_features, True)
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)


class TestMatrices:
    def test_euclidean_influence(self, sample_features):
        dist, infl = compute_distance_and_influence(sample_features, "euclidean")
        assert dist.shape == (12, 12)
        assert np.allclose(np.diag(infl), 0.0)
        i, j = 0, 1
        assert infl[i, j] == pytest.approx(1.0 / dist[i, j])

    def test_cosine_influence(self, sample_features):
        dist, infl = compute_distance_and_influence(sample_features, "cosine")
        assert np.allclose(np.diag(infl), 0.0)
        # Off-diagonal: influence = 1 - cosine distance (diag forced to 0).
        mask = ~np.eye(dist.shape[0], dtype=bool)
        assert np.allclose(infl[mask], (1.0 - dist)[mask])


class TestKNNRankings:
    def test_excludes_self(self, sample_features):
        dist, _ = compute_distance_and_influence(sample_features, "euclidean")
        knn = knn_rankings(dist, k=5)
        assert knn.shape == (12, 5)
        for i in range(12):
            assert i not in knn[i]

    def test_caps_at_n_minus_one(self, sample_features):
        dist, _ = compute_distance_and_influence(sample_features, "euclidean")
        knn = knn_rankings(dist, k=100)
        assert knn.shape == (12, 11)

    def test_requires_two_samples(self):
        dist = np.array([[0.0]])
        with pytest.raises(ValueError):
            knn_rankings(dist, k=1)
