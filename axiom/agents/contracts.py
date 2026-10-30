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

    paper_id: str
    action: Literal["full", "title_only", "skip"]
    reason: str


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["method", "task", "dataset", "metric"]
    evidence_span: str
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    entities: list[Entity]


class EnrichedEntity(Entity):
    canonical: str
    cosine: float = Field(ge=-1.0, le=1.0)
    ontology_source: Literal["openalex_concept", "concept_canonical", "unmatched"]


class EnrichmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    enriched: list[EnrichedEntity]


class VerifierError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    code: Literal["schema", "evidence_span", "ontology_membership", "cosine_threshold"]
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
