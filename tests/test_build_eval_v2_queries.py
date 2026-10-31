"""Offline tests for scripts/build_eval_v2_queries.py's grading/validation
logic (E3) -- never calls a real model; `llm_providers.complete_json` is
monkeypatched with a small stub queue, same pattern as
tests/test_agents_pipeline.py's FakeLLM.

Covers: `validate_judgment_coverage`'s duplicate/missing/extra-id checks, and
`grade_query`'s retry loop -- it must reject (and retry) a grading response
that doesn't cover the candidate pool exactly once each, and give up with a
clear error once retries are exhausted.
"""
from __future__ import annotations

import pytest

from axiom import llm_providers
from scripts.build_eval_v2_queries import (
    GradingResult,
    GradingValidationError,
    QueryJudgment,
    grade_query,
    validate_judgment_coverage,
)


def _judgment(paper_id: str, relevance: int = 1, reason: str = "on_topic") -> QueryJudgment:
    return QueryJudgment(paper_id=paper_id, relevance=relevance, reason=reason)


# ---------------------------------------------------------------------------
# validate_judgment_coverage
# ---------------------------------------------------------------------------
def test_validate_judgment_coverage_exact_match_is_valid():
    judgments = [_judgment("P1"), _judgment("P2")]
    assert validate_judgment_coverage(["P1", "P2"], judgments) == []


def test_validate_judgment_coverage_flags_duplicate():
    judgments = [_judgment("P1"), _judgment("P1"), _judgment("P2")]
    problems = validate_judgment_coverage(["P1", "P2"], judgments)
    assert any("duplicate" in p and "P1" in p for p in problems)


def test_validate_judgment_coverage_flags_missing():
    judgments = [_judgment("P1")]
    problems = validate_judgment_coverage(["P1", "P2"], judgments)
    assert any("missing" in p and "P2" in p for p in problems)


def test_validate_judgment_coverage_flags_extra_id_outside_pool():
    judgments = [_judgment("P1"), _judgment("P2"), _judgment("P99")]
    problems = validate_judgment_coverage(["P1", "P2"], judgments)
    assert any("not in the candidate pool" in p and "P99" in p for p in problems)


def test_validate_judgment_coverage_empty_pool_empty_judgments_is_valid():
    assert validate_judgment_coverage([], []) == []


# ---------------------------------------------------------------------------
# grade_query -- retry loop
# ---------------------------------------------------------------------------
class _StubGrader:
    """Programmable stand-in for `llm_providers.complete_json`: returns the
    next queued `GradingResult` (or raises, if an Exception was queued)."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def __call__(self, tier, system, prompt, model_cls, *, run_id, agent,
                        agent_type=None, paper_id=None, attempt=1, temperature=None,
                        client=None, conn=None):
        self.calls.append({"attempt": attempt, "prompt": prompt})
        if not self._responses:
            raise AssertionError("grade_query made more calls than responses were queued")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


_CANDIDATES = [
    {"paper_id": "P1", "title": "t1", "abstract": "a1"},
    {"paper_id": "P2", "title": "t2", "abstract": "a2"},
]


def test_grade_query_succeeds_first_attempt(monkeypatch):
    stub = _StubGrader([GradingResult(judgments=[_judgment("P1"), _judgment("P2")])])
    monkeypatch.setattr(llm_providers, "complete_json", stub)

    import asyncio
    result = asyncio.run(grade_query("q", _CANDIDATES, "run1"))
    assert {j.paper_id for j in result.judgments} == {"P1", "P2"}
    assert len(stub.calls) == 1


def test_grade_query_retries_after_incomplete_judgment_set_then_succeeds(monkeypatch):
    stub = _StubGrader([
        GradingResult(judgments=[_judgment("P1")]),  # missing P2 -- rejected, retried
        GradingResult(judgments=[_judgment("P1"), _judgment("P2")]),  # now covers the pool
    ])
    monkeypatch.setattr(llm_providers, "complete_json", stub)

    import asyncio
    result = asyncio.run(grade_query("q", _CANDIDATES, "run1", max_retries=3))
    assert {j.paper_id for j in result.judgments} == {"P1", "P2"}
    assert len(stub.calls) == 2
    # the retry prompt must mention the problem so the model can fix it
    assert "missing" in stub.calls[1]["prompt"]


def test_grade_query_retries_after_duplicate_then_succeeds(monkeypatch):
    stub = _StubGrader([
        GradingResult(judgments=[_judgment("P1"), _judgment("P1")]),  # duplicate, no P2
        GradingResult(judgments=[_judgment("P1"), _judgment("P2")]),
    ])
    monkeypatch.setattr(llm_providers, "complete_json", stub)

    import asyncio
    result = asyncio.run(grade_query("q", _CANDIDATES, "run1", max_retries=3))
    assert {j.paper_id for j in result.judgments} == {"P1", "P2"}
    assert len(stub.calls) == 2


def test_grade_query_exhausts_retries_raises_validation_error(monkeypatch):
    stub = _StubGrader([
        GradingResult(judgments=[_judgment("P1")]),
        GradingResult(judgments=[_judgment("P1")]),
    ])
    monkeypatch.setattr(llm_providers, "complete_json", stub)

    import asyncio
    with pytest.raises(GradingValidationError) as excinfo:
        asyncio.run(grade_query("q", _CANDIDATES, "run1", max_retries=2))
    assert "missing" in str(excinfo.value)
    assert len(stub.calls) == 2


def test_grade_query_propagates_provider_error_on_last_attempt(monkeypatch):
    stub = _StubGrader([
        llm_providers.LLMRefusalError("refused"),
        llm_providers.LLMRefusalError("refused again"),
    ])
    monkeypatch.setattr(llm_providers, "complete_json", stub)

    import asyncio
    with pytest.raises(llm_providers.LLMRefusalError):
        asyncio.run(grade_query("q", _CANDIDATES, "run1", max_retries=2))
    assert len(stub.calls) == 2
