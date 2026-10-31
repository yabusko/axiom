"""Tests for axiom.agents.contracts -- the Pydantic v2 boundary models (R2).

Mostly shape/constraint checks: `extra="forbid"`, the `ge`/`le` bounds that
the provider layer strips from the outbound JSON Schema but Pydantic still
enforces on the parsed response (the verifier's real backstop, OD23 #4).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from axiom.agents.contracts import (
    EnrichedEntity,
    Entity,
    EnrichmentResult,
    ExtractionResult,
    RouteDecision,
    RunStatusOut,
    VerifierError,
    VerifierResult,
)


def test_route_decision_valid():
    rd = RouteDecision(action="full", reason="has abstract")
    assert rd.action == "full"


def test_route_decision_route_source_defaults_to_rule():
    rd = RouteDecision(action="skip", reason="no title or abstract")
    assert rd.route_source == "rule"


def test_route_decision_accepts_llm_route_source():
    rd = RouteDecision(action="full", reason="x", route_source="llm")
    assert rd.route_source == "llm"


def test_route_decision_rejects_unknown_action():
    with pytest.raises(ValidationError):
        RouteDecision(action="maybe", reason="x")


def test_route_decision_rejects_extra_field():
    with pytest.raises(ValidationError):
        RouteDecision(action="full", reason="x", extra_field=1)


def test_route_decision_rejects_paper_id():
    """A4: paper_id is not a field on this LLM-facing contract at all -- an
    attempt to pass one (e.g. a stub mimicking a model that tried to echo
    one back) is rejected by extra="forbid", not silently accepted."""
    with pytest.raises(ValidationError):
        RouteDecision(paper_id="W1", action="full", reason="x")


def test_entity_confidence_bounds():
    Entity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.0)
    Entity(name="LoRA", kind="method", evidence_span="LoRA", confidence=1.0)
    with pytest.raises(ValidationError):
        Entity(name="LoRA", kind="method", evidence_span="LoRA", confidence=1.1)
    with pytest.raises(ValidationError):
        Entity(name="LoRA", kind="method", evidence_span="LoRA", confidence=-0.1)


def test_entity_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        Entity(name="LoRA", kind="algorithm", evidence_span="LoRA", confidence=0.5)


def test_extraction_result_roundtrip():
    result = ExtractionResult(
        entities=[Entity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9)],
    )
    assert result.entities[0].kind == "method"
    dumped = result.model_dump()
    assert "paper_id" not in dumped


def test_extraction_result_rejects_paper_id():
    """A4: the model is never asked for, and can never successfully return,
    a paper_id on this contract."""
    with pytest.raises(ValidationError):
        ExtractionResult(paper_id="W1", entities=[])


def test_enriched_entity_extends_entity_with_cosine_bounds():
    ee = EnrichedEntity(
        name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9,
        canonical="Low-Rank Adaptation", cosine=0.95, ontology_source="embedding",
    )
    assert ee.cosine == 0.95
    with pytest.raises(ValidationError):
        EnrichedEntity(
            name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9,
            canonical="x", cosine=1.5, ontology_source="embedding",
        )


@pytest.mark.parametrize("source", ["lexical", "embedding", "new_concept"])
def test_enriched_entity_accepts_each_renamed_ontology_source(source):
    ee = EnrichedEntity(
        name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9,
        canonical="x", cosine=0.5, ontology_source=source,
    )
    assert ee.ontology_source == source


def test_enriched_entity_rejects_pre_rename_ontology_source():
    """B2: the old openalex_concept|concept_canonical|unmatched vocabulary
    is gone -- a value from that set is now invalid, not just differently
    named."""
    with pytest.raises(ValidationError):
        EnrichedEntity(
            name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9,
            canonical="x", cosine=0.5, ontology_source="unmatched",
        )


def test_enriched_entity_rejects_unknown_ontology_source():
    with pytest.raises(ValidationError):
        EnrichedEntity(
            name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9,
            canonical="x", cosine=0.5, ontology_source="wikipedia",
        )


def test_enrichment_result_roundtrip():
    result = EnrichmentResult(
        paper_id="W1",
        enriched=[EnrichedEntity(
            name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9,
            canonical="Low-Rank Adaptation", cosine=0.95, ontology_source="embedding",
        )],
    )
    assert result.enriched[0].canonical == "Low-Rank Adaptation"


def test_verifier_error_and_result():
    err = VerifierError(field="entities[0].evidence_span", code="evidence_span", message="not found")
    result = VerifierResult(ok=False, attempt=2, errors=[err])
    assert result.ok is False
    assert result.errors[0].code == "evidence_span"


def test_verifier_result_defaults_to_empty_errors():
    result = VerifierResult(ok=True, attempt=1)
    assert result.errors == []


@pytest.mark.parametrize("code", ["name_span_mismatch", "empty_after_retry"])
def test_verifier_error_accepts_new_codes(code):
    err = VerifierError(field="x", code=code, message="m")
    assert err.code == code


def test_verifier_error_rejects_unknown_code():
    with pytest.raises(ValidationError):
        VerifierError(field="x", code="vibes", message="bad")


def test_run_status_out_from_attributes():
    class Row:
        run_id = "r1"
        status = "done"
        papers_total = 3
        papers_done = 3
        model_fast = "claude-haiku-4-5"
        model_strong = "claude-sonnet-5"
        started_at = "2026-10-30T00:00:00+00:00"
        finished_at = "2026-10-30T00:01:00+00:00"

    out = RunStatusOut.model_validate(Row(), from_attributes=True)
    assert out.run_id == "r1"
    assert out.status == "done"


def test_run_status_out_rejects_unknown_status():
    with pytest.raises(ValidationError):
        RunStatusOut(
            run_id="r1", status="exploding", papers_total=1, papers_done=0,
            model_fast="m", model_strong="m", started_at=None, finished_at=None,
        )
