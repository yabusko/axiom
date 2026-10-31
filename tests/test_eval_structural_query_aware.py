"""Offline tests for scripts/eval_structural_query_aware.py's pure helpers.

match_against_ontology/query_concepts/run need a real encoder, ontology, and
database -- only query_spans and query_aware_structural_scores (plain
dict/set logic) are exercised here.
"""
from __future__ import annotations

from scripts.eval_structural_query_aware import query_aware_structural_scores, query_spans


def test_query_spans_includes_whole_query_and_ngrams():
    spans = query_spans("retrieval augmented generation", max_n=2)
    assert "retrieval augmented generation" in spans
    assert "retrieval" in spans
    assert "augmented" in spans
    assert "generation" in spans
    assert "retrieval augmented" in spans
    assert "augmented generation" in spans
    assert "retrieval generation" not in spans  # not contiguous


def test_query_spans_longest_first():
    spans = query_spans("a b c", max_n=3)
    assert len(spans[0]) >= len(spans[-1])


def test_query_aware_structural_scores_counts_overlap(monkeypatch):
    class FakeConn:
        def execute(self, sql, params):
            class R:
                def fetchall(self_inner):
                    return [
                        {"paper_id": "P1", "canonical": "Retrieval"},
                        {"paper_id": "P1", "canonical": "Question answering"},
                        {"paper_id": "P2", "canonical": "Retrieval"},
                    ]
            return R()

    scores = query_aware_structural_scores(
        FakeConn(), ["P1", "P2"], {"Retrieval", "Question answering"}
    )
    assert scores["P1"] == 2.0  # shares both matched concepts
    assert scores["P2"] == 1.0  # shares only one


def test_query_aware_structural_scores_empty_concepts_returns_empty():
    class FakeConn:
        def execute(self, sql, params):
            raise AssertionError("should not query the db with no concepts to match")

    assert query_aware_structural_scores(FakeConn(), ["P1"], set()) == {}


def test_query_aware_structural_scores_empty_candidates_returns_empty():
    class FakeConn:
        def execute(self, sql, params):
            raise AssertionError("should not query the db with no candidates")

    assert query_aware_structural_scores(FakeConn(), [], {"x"}) == {}
