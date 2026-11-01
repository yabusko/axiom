"""Offline tests for scripts/judge_gaps.py's pure helpers.

The actual judging run makes real LLM calls against the live base corpus and
is not exercised here -- only prompt-building, cost estimation, top-paper
ranking, and report formatting.
"""
from __future__ import annotations

import networkx as nx

from axiom.gaps import Community, GapCandidate
from scripts import judge_gaps


def _graph():
    g = nx.DiGraph()
    g.add_node("P1", title="Paper One", year=2020, cited_by_count=50)
    g.add_node("P2", title="Paper Two", year=2021, cited_by_count=200)
    g.add_node("P3", title="Paper Three", year=2019, cited_by_count=5)
    return g


def _gap():
    a = Community(cid=0, members=["P1", "P2"], labels=["Dense passage retrieval"], size=2)
    b = Community(cid=1, members=["P3"], labels=["Hallucination"], size=1)
    return GapCandidate(a=a, b=b, semantic_similarity=0.99, inter_citations=3, gap_score=0.1, g_score=0.5)


def test_top_papers_ranks_by_cited_by_count_desc():
    g = _graph()
    ranked = judge_gaps._top_papers(g, ["P1", "P2", "P3"], top_k=2)
    assert [p.paper_id for p in ranked] == ["P2", "P1"]


def test_build_prompt_includes_topic_labels_and_citation_count():
    g = _graph()
    gap = _gap()
    prompt = judge_gaps.build_prompt("retrieval-augmented generation", gap, g)
    assert "retrieval-augmented generation" in prompt
    assert "Dense passage retrieval" in prompt
    assert "Hallucination" in prompt
    assert "3 time(s)" in prompt


def test_build_prompt_states_the_1_to_5_scale_explicitly():
    # Real run found the judge model once returned 55/35/45 instead of 5/3/4
    # (see judge_one's retry comment) -- the prompt must spell out the scale.
    prompt = judge_gaps.build_prompt("x", _gap(), _graph())
    assert "1 to 5" in prompt


def test_estimate_cost_scales_with_gap_count():
    small = judge_gaps.estimate_cost(1)
    big = judge_gaps.estimate_cost(10)
    assert small > 0
    assert big == small * 10


def test_write_report_aggregates_means_and_worth_pursuing(tmp_path, monkeypatch):
    out_path = tmp_path / "report.md"
    monkeypatch.setattr(judge_gaps, "OUT_REPORT", out_path)
    payload = {
        "generated_at": "2026-11-01T00:00:00Z",
        "spend_usd": 0.1,
        "_provenance": "test",
        "results": {
            "topic a": [
                {"gap_id": "abcdef0123456789", "a_label": "A", "b_label": "B", "inter_citations": 1,
                 "judgment": {"plausibility": 2, "novelty": 2, "actionability": 2, "worth_pursuing": False, "rationale": "r"}},
                {"gap_id": "fedcba9876543210", "a_label": "C", "b_label": "D", "inter_citations": 0,
                 "judgment": {"plausibility": 4, "novelty": 4, "actionability": 4, "worth_pursuing": True, "rationale": "r"}},
            ],
        },
    }
    judge_gaps.write_report(payload)
    text = out_path.read_text(encoding="utf-8")
    assert "Mean plausibility: 3.00 / 5" in text
    assert "Worth pursuing: 1/2 (50%)" in text
