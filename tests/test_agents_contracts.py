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
    rd = RouteDecision(paper_id="W1", action="full", reason="has abstract")
    assert rd.action == "full"


def test_route_decision_rejects_unknown_action():
    with pytest.raises(ValidationError):
        RouteDecision(paper_id="W1", action="maybe", reason="x")


def test_route_decision_rejects_extra_field():
    with pytest.raises(ValidationError):
        RouteDecision(paper_id="W1", action="full", reason="x", extra_field=1)


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
        paper_id="W1",
        entities=[Entity(name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9)],
    )
    assert result.entities[0].kind == "method"
    dumped = result.model_dump()
    assert dumped["paper_id"] == "W1"


def test_enriched_entity_extends_entity_with_cosine_bounds():
    ee = EnrichedEntity(
        name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9,
        canonical="Low-Rank Adaptation", cosine=0.95, ontology_source="concept_canonical",
    )
    assert ee.cosine == 0.95
    with pytest.raises(ValidationError):
        EnrichedEntity(
            name="LoRA", kind="method", evidence_span="LoRA", confidence=0.9,
            canonical="x", cosine=1.5, ontology_source="concept_canonical",
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
            canonical="Low-Rank Adaptation", cosine=0.95, ontology_source="concept_canonical",
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
