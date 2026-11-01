"""Pydantic v2 contracts at every agent boundary (R2).

Copied verbatim from `docs/agentic/ARCHITECTURE.md` §3 -- that document is the
frozen design; this module is its implementation, not a redesign. Pydantic v2
is the contract only at LLM/agent call boundaries (OD23 #4); the rest of
`axiom/*` keeps the OD12 dataclass convention.

`ge`/`le` constraints are enforced by Pydantic on the *parsed response*; the
provider layer (`axiom/llm_providers.py`) strips them from the JSON Schema
sent to the Anthropic API (unsupported keywords there) -- the verifier
(`axiom/agents/verifier.py`), not the API, is the backstop for those checks.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["full", "title_only", "skip"]
    reason: str
    # A5: which path produced this decision -- "rule" for the deterministic
    # checks in coordinator.route(), "llm" if a future ambiguous case ever
    # routes through the fast-tier model. Never model-supplied: coordinator.py
    # sets this itself, the same way pipeline.py owns paper_id (A4) rather
    # than trusting an LLM-facing field for it.
    route_source: Literal["rule", "llm"] = "rule"


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["method", "task", "dataset", "metric"]
    evidence_span: str
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # A4: no `paper_id` field here. This model is sent to the LLM as the
    # output schema (`llm_providers.complete_json(..., ExtractionResult)`);
    # the paper a result belongs to must never depend on the model echoing an
    # identifier back correctly. Callers (axiom/agents/pipeline.py) always
    # know which paper they're processing from their own input and attach it
    # downstream (EnrichmentResult.paper_id, linking.py's writes) themselves.
    entities: list[Entity]


class EnrichedEntity(Entity):
    canonical: str
    cosine: float = Field(ge=-1.0, le=1.0)
    # OD26/B1/B2: renamed from openalex_concept|concept_canonical|unmatched to
    # describe what actually produced the match rather than which table it
    # came from -- "lexical" (exact/normalized name match, step 1 of the
    # cascade), "embedding" (short-phrase-model cosine match, step 2), or
    # "new_concept" (no match found; kept with canonical = its own normalized
    # name, step 3 -- no longer dropped from linking).
    ontology_source: Literal["lexical", "embedding", "new_concept"]


class EnrichmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    enriched: list[EnrichedEntity]


class VerifierError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    # A3 adds "name_span_mismatch" (the claimed entity name doesn't relate to
    # its own evidence_span); A2 adds "empty_after_retry" (a paper-level
    # marker: at least one entity was proposed across extraction's attempts,
    # but none ever survived verification -- distinct from a paper that
    # legitimately has zero extractable entities, which is never an error).
    code: Literal[
        "schema", "evidence_span", "ontology_membership", "cosine_threshold",
        "name_span_mismatch", "empty_after_retry",
    ]
    message: str


class VerifierResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    attempt: int
    errors: list[VerifierError] = []


class RunStatusOut(BaseModel):
    """api/main.py response model, from_attributes=True (OD12 pattern)."""

    model_config = ConfigDict(from_attributes=True)

    run_id: str
    status: Literal["queued", "running", "done", "failed"]
    papers_total: int
    papers_done: int
    model_fast: str
    model_strong: str
    started_at: str | None
    finished_at: str | None
