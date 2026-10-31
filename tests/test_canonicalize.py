"""Tests for axiom.canonicalize — grouping, hallucination filter, manual skip.

Mocks axiom.llm_providers.complete_json directly (not the conftest-global
fake_llm fixture, which only stands in for axiom.llm.chat_json) -- stage-3
cleanup item A.1.6 moved this module off axiom/llm.py onto the shared
provider client.
"""
from __future__ import annotations

import pytest

from axiom import canonicalize, db, llm_providers as lp


class _FakeComplete:
    """Programmable stand-in for `llm_providers.complete_json`, queued in order."""

    def __init__(self):
        self.responses: list = []
        self.calls: list[str] = []

    async def __call__(self, tier, system, prompt, model_cls, *, agent,
                        agent_type=None, run_id="adhoc", paper_id=None,
                        attempt=1, temperature=None, client=None, conn=None):
        self.calls.append(prompt)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return model_cls(**item)


@pytest.fixture
def fake_llm(monkeypatch):
    fake = _FakeComplete()
    monkeypatch.setattr(lp, "complete_json", fake)
    return fake


def test_canonicalize_batch_maps_groups(fake_llm):
    fake_llm.responses = [{
        "groups": [
            {"canonical": "Low-Rank Adaptation", "members": ["LoRA", "Low-Rank Adaptation"]},
            {"canonical": "Machine translation", "members": ["Machine translation"]},
        ]
    }]
    result = canonicalize.canonicalize_batch(["LoRA", "Low-Rank Adaptation", "Machine translation"])
    assert result.mapping["LoRA"] == "Low-Rank Adaptation"
    assert result.mapping["Machine translation"] == "Machine translation"
    # only size>1 groups are surfaced for review.
    assert result.groups == [["LoRA", "Low-Rank Adaptation"]]


def test_hallucinated_members_filtered(fake_llm):
    # Model returns a member that wasn't in the input → dropped.
    fake_llm.responses = [{
        "groups": [{"canonical": "X", "members": ["X", "NOT_IN_INPUT"]}]
    }]
    result = canonicalize.canonicalize_batch(["X", "Y"])
    assert "NOT_IN_INPUT" not in result.mapping
    assert result.mapping["X"] == "X"
    assert result.mapping["Y"] == "Y"          # dropped label → identity fallback


def test_run_skips_manual_and_persists(empty_conn, fake_llm):
    for pid in ("W1", "W2"):
        db.insert_paper(empty_conn, openalex_id=pid, title=pid, abstract="a",
                        publication_year=2020, venue_id=None, venue="ACL",
                        cited_by_count=0, doi=None)
    db.insert_concepts(empty_conn, "W1", [("LoRA", 2)])
    db.insert_concepts(empty_conn, "W2", [("Machine translation", 1)])
    db.upsert_canonical(empty_conn, "LoRA", "LoRA", source="manual")
    empty_conn.commit()

    # Only "Machine translation" is unmapped; model echoes it back.
    fake_llm.responses = [{"groups": [{"canonical": "Machine translation",
                                       "members": ["Machine translation"]}]}]
    written = canonicalize.run(empty_conn, log=lambda *a, **k: None)
    assert written == 1                        # LoRA (manual) untouched
    assert "Machine translation" in fake_llm.calls[0]
    assert db.canonical_map(empty_conn)["LoRA"] == "LoRA"   # manual preserved
