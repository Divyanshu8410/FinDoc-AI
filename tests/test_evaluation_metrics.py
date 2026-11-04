import pytest

from evaluation.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    assert precision_at_k([10, 20, 30, 40], [20, 40], k=4) == 0.5


def test_recall_at_k():
    assert recall_at_k([10, 20, 30, 40], [20, 40, 50], k=4) == pytest.approx(2 / 3)


def test_precision_uses_actual_result_count():
    assert precision_at_k([10, 20], [20], k=10) == 0.5


def test_duplicate_retrieved_ids_are_counted_once():
    assert precision_at_k([10, 10, 20], [10], k=3) == 0.5


def test_empty_retrieval_returns_zero_precision_and_recall():
    assert precision_at_k([], [10], k=5) == 0.0
    assert recall_at_k([], [10], k=5) == 0.0


def test_empty_relevance_returns_zero_recall():
    assert recall_at_k([10, 20], [], k=2) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank([10, 20, 30], [30], k=3) == pytest.approx(1 / 3)


def test_invalid_k_raises_error():
    with pytest.raises(ValueError):
        precision_at_k([1], [1], k=0)
