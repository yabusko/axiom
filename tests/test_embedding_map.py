"""Tests for axiom.embedding_map — PCA projection, cosine, plain-language bands."""
from __future__ import annotations

from axiom import embedding_map as em


def test_project_preserves_ids_and_dims():
    vectors = {"A": [1.0, 0.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0, 0.0], "C": [1.0, 1.0, 0.0, 0.0]}
    coords = em.project(vectors, n_components=3)
    assert set(coords) == set(vectors)
    assert all(len(v) == 3 for v in coords.values())


def test_project_is_deterministic():
    vectors = {"A": [1.0, 2.0, 3.0], "B": [3.0, 1.0, 0.5], "C": [0.1, 0.2, 5.0]}
    assert em.project(vectors) == em.project(vectors)


def test_project_handles_fewer_than_two_papers():
    assert em.project({"A": [1.0, 2.0, 3.0]}) == {"A": [0.0, 0.0, 0.0]}
    assert em.project({}) == {}


def test_cosine_identical_and_orthogonal():
    assert em.cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert em.cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_zero_vector_is_zero():
    assert em.cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_explain_bands_are_monotonic_and_worded():
    assert "near-identical" in em.explain(0.95)
    assert "closely related" in em.explain(0.7)
    assert "loosely related" in em.explain(0.5)
    assert "distant" in em.explain(0.1)
    assert "%" in em.explain(0.5)
