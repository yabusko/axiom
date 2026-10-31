"""Tests for axiom.hypothesis — rule-based verifier + retry/error handling."""
from __future__ import annotations

import networkx as nx
import pytest

from axiom import hypothesis, llm
from axiom.gaps import Community, GapCandidate


def _gap_and_graph():
    g = nx.DiGraph()
    for pid, cites in (("A1", 10), ("A2", 5), ("B1", 8), ("B2", 4)):
        g.add_node(pid, title=f"title {pid}", year=2021, cited_by_count=cites, in_corpus=True)
    a = Community(cid=0, members=["A1", "A2"], labels=["retrieval"], size=2)
    b = Community(cid=1, members=["B1", "B2"], labels=["translation"], size=2)
    gap = GapCandidate(a=a, b=b, semantic_similarity=0.7, inter_citations=0, gap_score=0.7)
    return gap, g


def test_verify_requires_both_sides():
    # Two ids but both from side A → rejected.
    assert hypothesis.verify(
        {"supporting_paper_ids": ["A1", "A2"], "title": "t", "claim": "c",
         "method_sketch": "m"}, {"A1", "A2"}, {"B1", "B2"}) is None


def test_verify_requires_two_ids():
    assert hypothesis.verify(
        {"supporting_paper_ids": ["A1"], "title": "t", "claim": "c",
         "method_sketch": "m"}, {"A1"}, {"B1"}) is None


def test_verify_filters_unknown_ids_and_missing_fields():
    # Unknown id dropped; then only A1 + B1 remain → ok, but missing claim → None.
    assert hypothesis.verify(
        {"supporting_paper_ids": ["A1", "B1", "ZZ"], "title": "t",
         "method_sketch": "m"}, {"A1"}, {"B1"}) is None


def test_verify_success():
    pitch = hypothesis.verify(
        {"supporting_paper_ids": ["A1", "B1"], "title": "t", "claim": "c",
         "method_sketch": "m", "datasets": ["SQuAD", 5]}, {"A1"}, {"B1"})
    assert pitch is not None
    assert pitch.supporting_paper_ids == ["A1", "B1"]
    assert pitch.datasets == ["SQuAD"]         # non-str dataset filtered out


def test_explain_reports_insufficient_ids():
    reason = hypothesis.explain(
        {"supporting_paper_ids": ["A1"], "title": "t", "claim": "c", "method_sketch": "m"},
        {"A1", "A2"}, {"B1", "B2"},
    )
    assert ">=2 valid ids" in reason
    assert "no valid id from Side B" in reason


def test_explain_reports_missing_fields():
    reason = hypothesis.explain(
        {"supporting_paper_ids": ["A1", "B1"]}, {"A1"}, {"B1"},
    )
    assert "`title`" in reason
    assert "`claim`" in reason
    assert "`method_sketch`" in reason


def test_explain_success_case_has_no_reasons_but_returns_a_string():
    # explain() is only ever called after verify() returned None, but should
    # still degrade gracefully (not raise) if called on a passing payload.
    reason = hypothesis.explain(
        {"supporting_paper_ids": ["A1", "B1"], "title": "t", "claim": "c", "method_sketch": "m"},
        {"A1"}, {"B1"},
    )
    assert isinstance(reason, str) and reason


def test_generate_success_first_try(fake_llm):
    gap, g = _gap_and_graph()
    fake_llm.responses = [{
        "title": "Bridge", "claim": "c", "method_sketch": "m",
        "datasets": [], "supporting_paper_ids": ["A1", "B1"]}]
    pitch = hypothesis.generate_hypothesis(gap, g)
    assert pitch.title == "Bridge"
    assert len(fake_llm.calls) == 1


def test_generate_retries_then_succeeds(fake_llm):
    gap, g = _gap_and_graph()
    fake_llm.responses = [
        {"title": "t", "claim": "c", "method_sketch": "m", "supporting_paper_ids": ["A1"]},
        {"title": "t", "claim": "c", "method_sketch": "m", "supporting_paper_ids": ["A1", "B1"]},
    ]
    pitch = hypothesis.generate_hypothesis(gap, g)
    assert pitch.supporting_paper_ids == ["A1", "B1"]
    assert len(fake_llm.calls) == 2
    assert "previous attempt" in fake_llm.calls[1]   # nudge appended on retry


def test_generate_raises_verification_error(fake_llm):
    gap, g = _gap_and_graph()
    fake_llm.responses = [
        {"title": "t", "claim": "c", "method_sketch": "m", "supporting_paper_ids": ["A1"]}
    ] * hypothesis.MAX_RETRIES
    with pytest.raises(hypothesis.VerificationError):
        hypothesis.generate_hypothesis(gap, g)


def test_generate_reraises_ollama_error(fake_llm):
    gap, g = _gap_and_graph()
    fake_llm.responses = [llm.OllamaError("down")] * hypothesis.MAX_RETRIES
    with pytest.raises(llm.OllamaError):
        hypothesis.generate_hypothesis(gap, g)
