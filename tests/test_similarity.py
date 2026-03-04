import math

import pytest

from llm_semantic_cache.similarity import cosine_similarity


def test_identical_vectors_return_one() -> None:
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_return_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_return_negative_one() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_zero_vector_returns_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_mismatched_lengths_raise_value_error() -> None:
    with pytest.raises(ValueError, match="same length"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_result_is_float() -> None:
    assert isinstance(cosine_similarity([1.0, 1.0], [1.0, 0.0]), float)


def test_known_similarity_value() -> None:
    expected = 1 / math.sqrt(2)
    assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 1.0, 0.0]) == pytest.approx(expected)
