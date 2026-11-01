"""Tests for axiom.corpus_slice — relevance-floor field slicing (stage 4 A4/A5)."""
from __future__ import annotations

import networkx as nx

from axiom import corpus_slice


def _graph(edges):
    g = nx.DiGraph()
    g.add_edges_from(edges)
    return g


VECTORS = {
    "A": [1.0, 0.0, 0.0],   # cosine 1.0 to the query
    "B": [0.9, 0.1, 0.0],   # close
    "C": [0.0, 1.0, 0.0],   # orthogonal -> below any reasonable floor
    "N": [0.95, 0.05, 0.0],  # a citation neighbor of A, also close
}
QUERY = [1.0, 0.0, 0.0]


def test_slice_picks_papers_above_floor_only():
    g = _graph([])
    result = corpus_slice.compute_slice(g, VECTORS, QUERY, target_n=10, floor=0.8)
    assert "A" in result.paper_ids
    assert "B" in result.paper_ids
    assert "C" not in result.paper_ids


def test_slice_respects_target_n():
    g = _graph([])
    result = corpus_slice.compute_slice(g, VECTORS, QUERY, target_n=1, floor=0.0)
    # Only the top-1 direct hit ("A", cosine 1.0) makes it into the direct set.
    assert result.paper_ids[0] == "A"


def test_slice_adds_1hop_neighbors_that_also_clear_the_floor():
    g = _graph([("A", "N"), ("N", "C")])
    result = corpus_slice.compute_slice(g, VECTORS, QUERY, target_n=1, floor=0.8)
    # N is A's 1-hop neighbor and clears the floor -> included.
    assert "N" in result.paper_ids
    # C is N's neighbor (2-hop from A) and below the floor either way -> excluded.
    assert "C" not in result.paper_ids


def test_thin_flag_follows_min_papers(monkeypatch):
    from axiom import config
    monkeypatch.setattr(config, "FIELD_MIN_PAPERS", 4)
    g = _graph([])
    result = corpus_slice.compute_slice(g, VECTORS, QUERY, target_n=10, floor=0.8)
    assert result.total_relevant == 3  # A, B, N all clear 0.8 cosine
    assert result.thin is True


def test_empty_corpus_is_thin():
    result = corpus_slice.compute_slice(nx.DiGraph(), {}, QUERY, target_n=10, floor=0.8)
    assert result.paper_ids == []
    assert result.thin is True
