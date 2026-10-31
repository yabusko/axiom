"""Tests for axiom.agents.synthesis -- cluster labeling + hypothesis generation,
fully offline (axiom.llm_providers.complete_json monkeypatched). D4: this is the
one real hypothesis-generation implementation; axiom/hypothesis.py keeps only the
shared prompt/verify/explain helpers this module imports and reuses (see
tests/test_hypothesis.py), not its own generate_hypothesis anymore.
"""
from __future__ import annotations

import asyncio

import networkx as nx
import pytest

from axiom import llm_providers as lp
from axiom.agents import synthesis
from axiom.gaps import Community, GapCandidate
from axiom.hypothesis import VerificationError


class FakeLLM:
    """Programmable stand-in for `llm_providers.complete_json`, queued in order."""

    def __init__(self):
        self.responses: list = []
        self.calls: list[str] = []

    async def __call__(self, tier, system, prompt, model_cls, *, run_id, agent,
                        agent_type=None, paper_id=None, attempt=1, temperature=None,
                        client=None, conn=None):
        self.calls.append(prompt)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return model_cls(**item)


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(lp, "complete_json", fake)
    return fake


def _gap_and_graph():
    g = nx.DiGraph()
    for pid, cites in (("A1", 10), ("A2", 5), ("B1", 8), ("B2", 4)):
        g.add_node(pid, title=f"title {pid}", year=2021, cited_by_count=cites, in_corpus=True)
    a = Community(cid=0, members=["A1", "A2"], labels=["retrieval"], size=2)
    b = Community(cid=1, members=["B1", "B2"], labels=["translation"], size=2)
    gap = GapCandidate(a=a, b=b, semantic_similarity=0.7, inter_citations=0, gap_score=0.7)
    return gap, g


# --- label_clusters --------------------------------------------------------------

def test_label_clusters_empty_input_short_circuits(fake_llm):
    assert asyncio.run(synthesis.label_clusters([], run_id="r1")) == []
    assert fake_llm.calls == []  # no complete_json call at all


def test_label_clusters_success(fake_llm):
    clusters = [{"cluster_id": 0, "concepts": ["retrieval", "reranking"]}]
    fake_llm.responses = [{"labels": [{"cluster_id": 0, "label": "Retrieval"}]}]
    out = asyncio.run(synthesis.label_clusters(clusters, run_id="r1"))
    assert out == [{"cluster_id": 0, "label": "Retrieval"}]


def test_label_clusters_retries_then_raises_after_exhaustion(fake_llm):
    clusters = [{"cluster_id": 0, "concepts": ["x"]}]
    fake_llm.responses = [lp.LLMTransportError("down")] * 3
    with pytest.raises(lp.LLMTransportError):
        asyncio.run(synthesis.label_clusters(clusters, run_id="r1", max_retries=3))
    assert len(fake_llm.calls) == 3


# --- generate_hypothesis ---------------------------------------------------------

def test_generate_hypothesis_success_first_try(fake_llm):
    gap, g = _gap_and_graph()
    fake_llm.responses = [{
        "title": "Bridge", "claim": "c", "method_sketch": "m",
        "datasets": [], "supporting_paper_ids": ["A1", "B1"],
    }]
    pitch = asyncio.run(synthesis.generate_hypothesis(gap, g, run_id="r1"))
    assert pitch.title == "Bridge"
    assert len(fake_llm.calls) == 1


def test_generate_hypothesis_retries_with_nudge_then_succeeds(fake_llm):
    gap, g = _gap_and_graph()
    fake_llm.responses = [
        {"title": "t", "claim": "c", "method_sketch": "m", "supporting_paper_ids": ["A1"]},
        {"title": "t", "claim": "c", "method_sketch": "m", "supporting_paper_ids": ["A1", "B1"]},
    ]
    pitch = asyncio.run(synthesis.generate_hypothesis(gap, g, run_id="r1"))
    assert pitch.supporting_paper_ids == ["A1", "B1"]
    assert len(fake_llm.calls) == 2
    assert "previous attempt" in fake_llm.calls[1]  # structured nudge, not a generic reminder


def test_generate_hypothesis_raises_verification_error_after_exhaustion(fake_llm):
    gap, g = _gap_and_graph()
    fake_llm.responses = [
        {"title": "t", "claim": "c", "method_sketch": "m", "supporting_paper_ids": ["A1"]}
    ] * 3
    with pytest.raises(VerificationError):
        asyncio.run(synthesis.generate_hypothesis(gap, g, run_id="r1", max_retries=3))


def test_generate_hypothesis_reraises_provider_error_when_nothing_ever_produced(fake_llm):
    gap, g = _gap_and_graph()
    fake_llm.responses = [lp.LLMTransportError("down")] * 3
    with pytest.raises(lp.LLMTransportError):
        asyncio.run(synthesis.generate_hypothesis(gap, g, run_id="r1", max_retries=3))
