"""Tests for axiom.hypothesis — shared prompt/verify/explain helpers reused by
axiom.agents.synthesis.generate_hypothesis, the one real hypothesis-generation
implementation (D4; see that module's generate_hypothesis for retry/error-handling
coverage over the full call, which now runs through axiom.llm_providers)."""
from __future__ import annotations

from axiom import hypothesis


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
