"""Extraction agent (R1/R4): fast-tier call that pulls method/task/dataset/
metric entities out of a paper, each grounded in a verbatim `evidence_span`.

Wrapped in the verifier retry loop (R3/R4): after every call, `verifier.
verify_extraction` checks `evidence_span` is actually a substring of the
title/abstract and that `name` relates to its own span (A1/A3). On failure
the error list is rendered back into the next prompt and the SAME agent
retries, up to `config.AGENT_MAX_RETRIES` attempts. A `pydantic.
ValidationError` (schema check) is treated the same way -- its `.errors()`
become `VerifierError`s via `verifier.errors_from_validation`.

A2: verdicts are tracked PER ENTITY, accumulated across attempts, not as one
pass/fail for the whole call. An entity that passes verification on any
attempt is kept permanently, even if a later attempt's response drops it or
proposes something else that fails; only entities that NEVER pass across
every attempt are left out of the final result. The whole paper is only
rejected (via `verifier.VerificationFailed`) if EVERY entity ever proposed
failed every attempt -- a response that mixes good and bad entities no
longer drags the good ones down with it. If no entity was ever proposed at
all (the model legitimately found nothing, every attempt returned an empty
or all-still-empty list with no verifier errors), that's success with zero
entities, not a rejection -- a paper can genuinely have nothing to extract.
`empty_after_retry` on the raised exception's error list is the explicit
marker for the other case: something WAS proposed, but nothing ever survived.

A4: this module never asks the model for `paper_id` and never trusts one
back -- `ExtractionResult` has no `paper_id` field (contracts.py); the
caller (`axiom/agents/pipeline.py`) always knows which paper it's processing
from its own input, not from anything parsed out of the model's response.

A5: `mode="title_only"` (set by the coordinator's rule-based route for a
paper with no usable abstract) uses a distinct prompt that doesn't mention
an abstract at all, and verifies spans against the title only -- passing
`abstract=None` to `verify_extraction` regardless of whatever (too-thin)
abstract text the caller may still be holding, since the route decision
already judged it unusable.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import httpx
from pydantic import ValidationError

from axiom import config, llm_providers
from axiom.agents import verifier
from axiom.agents.contracts import Entity, ExtractionResult, VerifierError

_SYSTEM_FULL = (
    "You extract research entities from a paper's title and abstract. Entity "
    f"kinds: {', '.join(config.ENTITY_KINDS)}. For every entity, `evidence_span` "
    "MUST be copied VERBATIM (exact substring, same characters, same case) from "
    "the title or abstract text given -- never paraphrase it. `name` must "
    "clearly relate to the evidence_span you cite for it. `confidence` is your "
    "belief the entity is correctly identified, 0.0-1.0. Only extract entities "
    "that are explicitly present in the text; never invent one."
)

_SYSTEM_TITLE_ONLY = (
    "You extract research entities from ONLY a paper's title -- no abstract is "
    f"available. Entity kinds: {', '.join(config.ENTITY_KINDS)}. For every "
    "entity, `evidence_span` MUST be copied VERBATIM (exact substring, same "
    "characters, same case) from the title given -- never paraphrase it, and "
    "never invent text that isn't in the title. `name` must clearly relate to "
    "the evidence_span you cite for it. `confidence` is your belief the entity "
    "is correctly identified, 0.0-1.0. A bare title usually names at most one "
    "or two entities; return an empty list if nothing is clearly identifiable."
)

_JSON_SHAPE = (
    'Return JSON: {"entities": [{"name": "...", '
    '"kind": "method"|"task"|"dataset"|"metric", "evidence_span": "...", '
    '"confidence": 0.0}]}. `entities` may be an empty list if none are '
    "clearly present."
)


def _prompt_full(title: str | None, abstract: str | None) -> str:
    return f"title: {title or ''}\nabstract: {abstract or ''}\n\n{_JSON_SHAPE}"


def _prompt_title_only(title: str | None) -> str:
    return f"title: {title or ''}\n\n{_JSON_SHAPE}"


@dataclass
class ExtractionOutcome:
    """`extract()`'s real return value -- `ExtractionResult` plus the
    bookkeeping pipeline.py needs (D3: real attempt count on the success
    path too, not just on failure)."""

    paper_id: str
    entities: list[Entity] = field(default_factory=list)
    attempts: int = 0


async def extract(
    paper_id: str,
    title: str | None,
    abstract: str | None,
    *,
    run_id: str,
    max_retries: int | None = None,
    mode: str = "full",
    client: httpx.AsyncClient | None = None,
    conn: sqlite3.Connection | None = None,
) -> ExtractionOutcome:
    """Extract + verify, retrying the SAME agent up to `max_retries` times,
    accumulating passing entities across attempts (A2 -- see module docstring).

    `max_retries` defaults to `config.AGENT_MAX_RETRIES` -- read at CALL time
    (not bound into the signature at import time) so a runtime change to the
    config value takes effect immediately.

    `mode`: `"full"` (title+abstract, default) or `"title_only"` (A5).

    `client`/`conn` (C2/C3): pass the run's pooled `httpx.AsyncClient` and a sqlite3
    connection through to `complete_json` so every call in a run shares one HTTP
    connection and reads the durable, cross-run spend total instead of each call
    opening its own client and checking only this process's in-memory total.

    Raises `verifier.VerificationFailed` if nothing ever survives verification
    AND at least one entity was proposed at some point (the `empty_after_retry`
    marker is attached to that exception's error list). A provider-layer
    failure that leaves no parseable result on the final attempt, with
    nothing accepted yet, re-raises the underlying exception instead."""
    max_retries = max_retries if max_retries is not None else config.AGENT_MAX_RETRIES
    title_only = mode == "title_only"
    system = _SYSTEM_TITLE_ONLY if title_only else _SYSTEM_FULL
    base_prompt = _prompt_title_only(title) if title_only else _prompt_full(title, abstract)
    verify_abstract = None if title_only else abstract

    accepted: dict[tuple, Entity] = {}
    ever_proposed = 0
    last_errors: list[VerifierError] = []
    last_provider_exc: Exception | None = None
    attempts_used = 0

    for attempt in range(1, max_retries + 1):
        attempts_used = attempt
        prompt = base_prompt
        if last_errors:
            prompt = f"{base_prompt}\n\n{verifier.format_errors_for_retry(last_errors)}"

        try:
            result = await llm_providers.complete_json(
                "fast", system, prompt, ExtractionResult,
                run_id=run_id, agent="extraction", agent_type="extraction",
                paper_id=paper_id, attempt=attempt, client=client, conn=conn,
            )
        except ValidationError as exc:
            last_errors = verifier.errors_from_validation(exc)
            last_provider_exc = exc
            verifier.log_attempt(
                run_id, "extraction",
                verifier.VerifierResult(ok=False, attempt=attempt, errors=last_errors),
                model=config.MODEL_FAST, paper_id=paper_id,
            )
            continue
        except llm_providers.LLMError as exc:
            if attempt >= max_retries:
                if accepted:
                    break
                raise
            last_provider_exc = exc
            last_errors = []
            continue

        ever_proposed += len(result.entities)
        verdict = verifier.verify_extraction(
            result, title=title, abstract=verify_abstract, attempt=attempt,
        )
        verifier.log_attempt(run_id, "extraction", verdict, model=config.MODEL_FAST, paper_id=paper_id)

        failing = {int(e.field.split("[", 1)[1].split("]", 1)[0]) for e in verdict.errors}
        for i, entity in enumerate(result.entities):
            if i in failing:
                continue
            key = (entity.name, entity.kind, entity.evidence_span)
            accepted.setdefault(key, entity)

        if verdict.ok:
            last_errors = []
            break
        last_errors = verdict.errors
        last_provider_exc = None

    final_entities = list(accepted.values())
    if final_entities:
        return ExtractionOutcome(paper_id=paper_id, entities=final_entities, attempts=attempts_used)

    if ever_proposed == 0:
        if last_provider_exc is not None:
            raise last_provider_exc
        # Legitimately nothing to extract -- not a failure.
        return ExtractionOutcome(paper_id=paper_id, entities=[], attempts=attempts_used)

    # Something was proposed at some point; nothing ever survived (A2).
    marker = VerifierError(
        field="entities",
        code="empty_after_retry",
        message=(
            f"{ever_proposed} entity attempt(s) proposed across {attempts_used} "
            "attempt(s) but none passed verification"
        ),
    )
    raise verifier.VerificationFailed("extraction", attempts_used, last_errors + [marker])
