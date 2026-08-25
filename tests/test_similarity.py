"""Tests for RBO and Jaccard."""

import numpy as np
import pytest

from bridg.similarity import compute_jaccard, compute_rbo, get_similarity_fn


class TestJaccard:
    def test_identical_lists(self):
        x = [0, 1, 2, 3]
        assert compute_jaccard(x, x, 4) == pytest.approx(1.0)

    def test_disjoint_lists(self):
        x = [0, 1, 2]
        y = [3, 4, 5]
        assert compute_jaccard(x, y, 3) == 0.0

    def test_partial_overlap(self):
        x = [0, 1, 2]
        y = [1, 2, 3]
        # intersection 2, union 4
        assert compute_jaccard(x, y, 3) == pytest.approx(0.5)

    def test_respects_top_k(self):
        x = [0, 1, 2, 3]
        y = [0, 9, 8, 7]
        assert compute_jaccard(x, y, 1) == pytest.approx(1.0)
        assert compute_jaccard(x, y, 2) == pytest.approx(1.0 / 3.0)


class TestRBO:
    def test_identical_lists(self):
        # Truncated residual RBO without extrapolation is < 1 even for ties.
        x = [0, 1, 2, 3, 4]
        score = compute_rbo(x, x, 5, p=0.9)
        expected = (1 - 0.9) * sum((0.9 ** i) * 1.0 for i in range(5))
        assert score == pytest.approx(expected)
        assert score > 0.0

    def test_disjoint_lists(self):
        x = [0, 1, 2]
        y = [3, 4, 5]
        assert compute_rbo(x, y, 3, p=0.9) == 0.0

    def test_truncated_depth(self):
        # Only first position matches; deeper ranks differ.
        x = [0, 1, 2, 3]
        y = [0, 9, 8, 7]
        full = compute_rbo(x, y, 4, p=0.9)
        shallow = compute_rbo(x, y, 1, p=0.9)
        assert shallow == pytest.approx((1 - 0.9) * 1.0)
        assert full > shallow

    def test_get_similarity_fn_rbo_p(self):
        fn = get_similarity_fn("rbo", rbo_p=0.9)
        x = [0, 1, 2]
        assert fn(x, x, 3) == pytest.approx(compute_rbo(x, x, 3, p=0.9))
