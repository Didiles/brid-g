"""Tests for ranking and self-exclusion."""

import numpy as np
import pytest
from scipy.spatial.distance import braycurtis, canberra, jensenshannon
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances

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

    @pytest.mark.parametrize("metric", ["manhattan", "chebyshev", "minkowski"])
    def test_lp_influence_is_inverse_distance(self, sample_features, metric):
        dist, infl = compute_distance_and_influence(sample_features, metric)
        assert np.allclose(np.diag(infl), 0.0)
        mask = ~np.eye(dist.shape[0], dtype=bool)
        assert np.allclose(infl[mask], 1.0 / dist[mask])

    def test_minkowski_uses_p(self, sample_features):
        dist, infl = compute_distance_and_influence(
            sample_features, "minkowski", minkowski_p=4
        )
        expected = pairwise_distances(sample_features, metric="minkowski", p=4)
        assert np.allclose(dist, expected)
        mask = ~np.eye(dist.shape[0], dtype=bool)
        assert np.allclose(infl[mask], 1.0 / dist[mask])

    def test_correlation_influence_is_pearson(self):
        features = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [3.0, 2.0, 1.0],
                [1.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
        dist, infl = compute_distance_and_influence(features, "correlation")
        pearson = np.corrcoef(features)
        mask = ~np.eye(features.shape[0], dtype=bool)
        assert np.allclose(np.diag(infl), 0.0)
        assert np.allclose(infl[mask], pearson[mask])
        assert np.allclose(dist[mask], (1.0 - pearson)[mask])

    def test_correlation_knn_follows_highest_pearson(self):
        features = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [3.0, 2.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist, infl = compute_distance_and_influence(features, "correlation")
        knn = knn_rankings(dist, k=1)
        assert knn[0, 0] == 1
        assert infl[0, 1] > infl[0, 2]

    def test_correlation_constant_row_rejected(self):
        features = np.array(
            [[1.0, 1.0, 1.0], [0.0, 1.0, 2.0], [1.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        with pytest.raises(ValueError, match="correlation"):
            compute_distance_and_influence(features, "correlation")

    def test_canberra_influence_is_inverse_distance(self, sample_features):
        dist, infl = compute_distance_and_influence(sample_features, "canberra")
        mask = ~np.eye(dist.shape[0], dtype=bool)
        assert np.allclose(np.diag(infl), 0.0)
        assert np.allclose(infl[mask], 1.0 / dist[mask])
        assert dist[0, 1] == pytest.approx(
            canberra(sample_features[0], sample_features[1])
        )

    def test_braycurtis_influence_is_inverse_distance(self):
        rng = np.random.RandomState(1)
        features = rng.rand(8, 4) + 0.1
        dist, infl = compute_distance_and_influence(features, "braycurtis")
        mask = ~np.eye(dist.shape[0], dtype=bool)
        assert np.allclose(np.diag(infl), 0.0)
        assert np.allclose(infl[mask], 1.0 / dist[mask])
        assert dist[0, 1] == pytest.approx(braycurtis(features[0], features[1]))

    def test_mahalanobis_influence_is_inverse_distance(self, sample_features):
        dist, infl = compute_distance_and_influence(sample_features, "mahalanobis")
        cov = np.cov(sample_features, rowvar=False, ddof=1)
        vi = np.linalg.inv(cov)
        delta = sample_features[0] - sample_features[1]
        expected = np.sqrt(delta @ vi @ delta)
        mask = ~np.eye(dist.shape[0], dtype=bool)
        assert np.allclose(np.diag(infl), 0.0)
        assert np.allclose(infl[mask], 1.0 / dist[mask])
        assert dist[0, 1] == pytest.approx(expected)

    def test_seuclidean_influence_is_inverse_distance(self, sample_features):
        dist, infl = compute_distance_and_influence(sample_features, "seuclidean")
        variances = np.var(sample_features, axis=0, ddof=1)
        delta = sample_features[0] - sample_features[1]
        expected = np.sqrt(np.sum(delta**2 / variances))
        mask = ~np.eye(dist.shape[0], dtype=bool)
        assert np.allclose(np.diag(infl), 0.0)
        assert np.allclose(infl[mask], 1.0 / dist[mask])
        assert dist[0, 1] == pytest.approx(expected)

    def test_jensenshannon_influence_is_inverse_distance(self):
        rng = np.random.RandomState(2)
        features = rng.rand(6, 4) + 0.1
        dist, infl = compute_distance_and_influence(features, "jensenshannon")
        probs = features / features.sum(axis=1, keepdims=True)
        mask = ~np.eye(dist.shape[0], dtype=bool)
        assert np.allclose(np.diag(infl), 0.0)
        assert np.allclose(infl[mask], 1.0 / dist[mask])
        assert dist[0, 1] == pytest.approx(jensenshannon(probs[0], probs[1]))

    def test_hellinger_influence_is_inverse_distance(self):
        rng = np.random.RandomState(3)
        features = rng.rand(6, 4) + 0.1
        dist, infl = compute_distance_and_influence(features, "hellinger")
        probs = features / features.sum(axis=1, keepdims=True)
        expected = np.linalg.norm(np.sqrt(probs[0]) - np.sqrt(probs[1])) / np.sqrt(2.0)
        mask = ~np.eye(dist.shape[0], dtype=bool)
        assert np.allclose(np.diag(infl), 0.0)
        assert np.allclose(infl[mask], 1.0 / dist[mask])
        assert dist[0, 1] == pytest.approx(expected)
        assert 0.0 <= dist[0, 1] <= 1.0

    def test_spearman_influence_is_coefficient(self):
        features = np.array(
            [
                [1.0, 1.0, 3.0],
                [2.0, 2.0, 5.0],
                [3.0, 0.0, 1.0],
                [0.0, 4.0, 2.0],
            ],
            dtype=np.float64,
        )
        dist, infl = compute_distance_and_influence(features, "spearman")
        rho = np.corrcoef(rankdata(features, method="average", axis=1))
        mask = ~np.eye(features.shape[0], dtype=bool)
        assert np.allclose(np.diag(infl), 0.0)
        assert np.allclose(infl[mask], rho[mask])
        assert np.allclose(dist[mask], (1.0 - rho)[mask])

    def test_spearman_knn_follows_highest_rho(self):
        features = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [3.0, 2.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist, infl = compute_distance_and_influence(features, "spearman")
        knn = knn_rankings(dist, k=1)
        assert knn[0, 0] == 1
        assert infl[0, 1] > infl[0, 2]

    def test_mahalanobis_singular_covariance_rejected(self):
        features = np.array(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=np.float64,
        )
        with pytest.raises(ValueError, match="mahalanobis"):
            compute_distance_and_influence(features, "mahalanobis")

    def test_seuclidean_zero_variance_rejected(self):
        features = np.array(
            [[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
            dtype=np.float64,
        )
        with pytest.raises(ValueError, match="seuclidean"):
            compute_distance_and_influence(features, "seuclidean")

    @pytest.mark.parametrize("metric", ["braycurtis", "jensenshannon", "hellinger"])
    def test_nonnegative_metrics_reject_negative_features(self, metric):
        features = np.array(
            [[1.0, -0.2], [0.4, 0.6], [0.2, 0.3]],
            dtype=np.float64,
        )
        with pytest.raises(ValueError, match=metric):
            compute_distance_and_influence(features, metric)

    @pytest.mark.parametrize("metric", ["jensenshannon", "hellinger"])
    def test_probability_metrics_reject_zero_row(self, metric):
        features = np.array(
            [[0.0, 0.0], [0.2, 0.8], [0.5, 0.5]],
            dtype=np.float64,
        )
        with pytest.raises(ValueError, match=metric):
            compute_distance_and_influence(features, metric)

    def test_braycurtis_zero_denominator_rejected(self):
        features = np.array(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 2.0]],
            dtype=np.float64,
        )
        with pytest.raises(ValueError, match="braycurtis"):
            compute_distance_and_influence(features, "braycurtis")

    def test_spearman_constant_row_rejected(self):
        features = np.array(
            [[1.0, 1.0, 1.0], [0.0, 1.0, 2.0], [1.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        with pytest.raises(ValueError, match="spearman"):
            compute_distance_and_influence(features, "spearman")


class TestKNNRankings:
    def test_excludes_self(self, sample_features):
        dist, _ = compute_distance_and_influence(sample_features, "euclidean")
        knn = knn_rankings(dist, k=5)
        assert knn.shape == (12, 5)
        for i in range(12):
            assert i not in knn[i]

    def test_excludes_self_with_duplicate_rows(self):
        features = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float64,
        )
        dist, _ = compute_distance_and_influence(features, "euclidean")
        knn = knn_rankings(dist, k=2)
        assert knn.shape == (3, 2)
        for i in range(3):
            assert i not in knn[i]

    def test_caps_at_n_minus_one(self, sample_features):
        dist, _ = compute_distance_and_influence(sample_features, "euclidean")
        knn = knn_rankings(dist, k=100)
        assert knn.shape == (12, 11)

    def test_requires_two_samples(self):
        dist = np.array([[0.0]])
        with pytest.raises(ValueError):
            knn_rankings(dist, k=1)
