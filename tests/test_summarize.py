"""Tests for axiom.summarize -- no prior coverage existed for this module.

Mocks axiom.llm_providers.complete_json directly, same pattern as
tests/test_canonicalize.py (stage-3 cleanup item A.1.6 moved both modules
off axiom/llm.py onto the shared provider client).
"""
from __future__ import annotations

import pytest

from axiom import llm_providers as lp
from axiom import summarize


class _FakeComplete:
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


def test_summarize_paper_returns_bullets(fake_llm):
    fake_llm.responses = [{"bullets": ["a", "b", "c"]}]
    result = summarize.summarize_paper("P1", "Title", "Abstract text.")
    assert result.paper_id == "P1"
    assert result.bullets == ["a", "b", "c"]
    assert "Abstract text." in fake_llm.calls[0]


def test_summarize_paper_truncates_to_three(fake_llm):
    fake_llm.responses = [{"bullets": ["a", "b", "c", "d"]}]
    result = summarize.summarize_paper("P1", "Title", "Abstract text.")
    assert result.bullets == ["a", "b", "c"]


def test_summarize_paper_drops_empty_bullets(fake_llm):
    fake_llm.responses = [{"bullets": ["", "a", ""]}]
    result = summarize.summarize_paper("P1", "Title", "Abstract text.")
    assert result.bullets == ["a"]


def test_summarize_paper_raises_on_no_usable_bullets(fake_llm):
    fake_llm.responses = [{"bullets": []}]
    with pytest.raises(lp.LLMParseError):
        summarize.summarize_paper("P1", "Title", "Abstract text.")


def test_summarize_paper_raises_on_no_abstract():
    with pytest.raises(ValueError):
        summarize.summarize_paper("P1", "Title", None)


def test_summarize_paper_propagates_llm_error(fake_llm):
    fake_llm.responses = [lp.LLMTransportError("unreachable")]
    with pytest.raises(lp.LLMTransportError):
        summarize.summarize_paper("P1", "Title", "Abstract text.")
