"""Enrichment agent (R1/R4): deterministic + SPECTER2 embeddings, NO LLM call.

Maps each extracted `Entity.name` to the ontology -- `concept_canonical`
rows union OpenAlex-sourced `concepts` rows -- by cosine similarity, via the
existing `axiom.embed.Specter2Encoder` interface (never reimplemented here;
the encoder is injected by the caller so it's constructed once per process,
not once per paper).

Still sits inside the same attempt<=3 verifier retry box as extraction
(ARCHITECTURE.md's data-flow diagram), but since this agent has no prompt to
nudge, "self-correction" means something else here: an entity that fails the
ontology-membership or cosine-threshold check is RE-EMBEDDED on the next
attempt with progressively more context (bare name -> name+kind ->
name+evidence_span) rather than re-ranked against the same fixed embedding --
re-ranking the same vector would only ever surface a *lower*-cosine
candidate, since the first pick is already the arg-max. Richer context can
genuinely shift the entity vector toward (or away from) a better ontology
match, which is the only lever a deterministic agent has.
"""
from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass

import numpy as np

from axiom import config
from axiom.agents import verifier
from axiom.agents.contracts import EnrichedEntity, Entity, EnrichmentResult, ExtractionResult, VerifierError


@dataclass
class Ontology:
    """Flat candidate pool: `concept_canonical.canonical` union `concepts.concept`,
    each row's SPECTER2 embedding pre-normalized (unit L2 norm) so cosine
    similarity is a plain dot product."""

    labels: list[str]
    sources: list[str]           # "concept_canonical" | "openalex_concept", parallel to labels
    vectors: np.ndarray           # (N, dim), L2-normalized rows; N may be 0


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_ontology(conn: sqlite3.Connection, encoder) -> Ontology:
    """Build the candidate pool once per pipeline run (not per paper/entity).
    Raw SQL against the existing tables -- `axiom/db.py` has no helper for
    this read, and this module doesn't own `db.py` to add one."""
    canonical_rows = conn.execute(
        "SELECT DISTINCT canonical FROM concept_canonical WHERE canonical IS NOT NULL"
    ).fetchall()
    concept_rows = conn.execute(
        "SELECT DISTINCT concept FROM concepts WHERE concept IS NOT NULL"
    ).fetchall()

    canonical_labels = sorted({r["canonical"] for r in canonical_rows if r["canonical"]})
    seen = set(canonical_labels)
    concept_labels = sorted({r["concept"] for r in concept_rows if r["concept"]} - seen)

    labels = canonical_labels + concept_labels
    sources = (["concept_canonical"] * len(canonical_labels)
               + ["openalex_concept"] * len(concept_labels))

    if not labels:
        return Ontology(labels=[], sources=[], vectors=np.zeros((0, 1)))

    vectors = _normalize_rows(np.array(encoder.encode(labels), dtype=float))
    return Ontology(labels=labels, sources=sources, vectors=vectors)


def _entity_text(entity: Entity, attempt: int) -> str:
    """Progressively richer context per retry -- see module docstring."""
    if attempt <= 1:
        return entity.name
    if attempt == 2:
        return f"{entity.name} ({entity.kind})"
    return f"{entity.name}: {entity.evidence_span}"


def _failed_indices(errors: list[VerifierError]) -> list[int]:
    """`enriched[3].cosine` -> 3, for every ontology_membership/cosine_threshold error."""
    out: set[int] = set()
    for err in errors:
        if err.code not in ("ontology_membership", "cosine_threshold"):
            continue
        field = err.field
        start = field.find("[")
        end = field.find("]")
        if start != -1 and end > start:
            try:
                out.add(int(field[start + 1:end]))
            except ValueError:
                continue
    return sorted(out)


def _unmatched(entity: Entity) -> EnrichedEntity:
    return EnrichedEntity(
        **entity.model_dump(), canonical="", cosine=-1.0, ontology_source="unmatched",
    )


async def enrich(
    extraction: ExtractionResult,
    *,
    encoder,
    ontology: Ontology,
    run_id: str,
    max_retries: int | None = None,
) -> EnrichmentResult:
    """Enrich every entity in `extraction`, verify, retry (same agent, richer
    embedding context each time) up to `max_retries` (defaults to
    `config.AGENT_MAX_RETRIES`, read at call time -- see extraction.py's
    `extract()` for why this isn't a bound default). Raises
    `verifier.VerificationFailed` if no attempt passes."""
    max_retries = max_retries if max_retries is not None else config.AGENT_MAX_RETRIES
    entities = extraction.entities
    enriched: list[EnrichedEntity | None] = [None] * len(entities)
    pending = list(range(len(entities)))
    last_errors: list[VerifierError] = []

    for attempt in range(1, max_retries + 1):
        if pending and ontology.vectors.shape[0] == 0:
            for idx in pending:
                enriched[idx] = _unmatched(entities[idx])
            pending = []
        elif pending:
            texts = [_entity_text(entities[idx], attempt) for idx in pending]
            raw_vectors = await asyncio.to_thread(encoder.encode, texts)
            vectors = _normalize_rows(np.array(raw_vectors, dtype=float))
            sims = vectors @ ontology.vectors.T
            best = sims.argmax(axis=1)
            for pos, entity_idx in enumerate(pending):
                chosen = int(best[pos])
                # float64 dot products of near-identical unit vectors can land
                # a hair past +-1.0 (e.g. 1.0000000000000004); clamp before
                # the strict ge=-1/le=1 Pydantic field raises on what is, in
                # cosine-similarity terms, exactly 1.0.
                cosine = max(-1.0, min(1.0, float(sims[pos, chosen])))
                enriched[entity_idx] = EnrichedEntity(
                    **entities[entity_idx].model_dump(),
                    canonical=ontology.labels[chosen],
                    cosine=cosine,
                    ontology_source=ontology.sources[chosen],
                )

        result = EnrichmentResult(paper_id=extraction.paper_id, enriched=list(enriched))  # type: ignore[arg-type]
        verdict = verifier.verify_enrichment(result, attempt=attempt)
        verifier.log_attempt(run_id, "enrichment", verdict, paper_id=extraction.paper_id)
        if verdict.ok:
            return result
        last_errors = verdict.errors
        pending = _failed_indices(last_errors)
        if not pending:
            # Verifier found errors but none map to a re-embeddable entity
            # (shouldn't happen given verify_enrichment's two codes) -- stop.
            break

    raise verifier.VerificationFailed("enrichment", max_retries, last_errors)
