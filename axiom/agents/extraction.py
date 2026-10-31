"""Extraction agent (R1/R4): fast-tier call that pulls method/task/dataset/
metric entities out of a paper, each grounded in a verbatim `evidence_span`.

Wrapped in the verifier retry loop (R3/R4): after every call, `verifier.
verify_extraction` checks `evidence_span` is actually a substring of the
abstract. On failure the error list is rendered back into the next prompt
and the SAME agent retries, up to `config.AGENT_MAX_RETRIES` attempts. A
`pydantic.ValidationError` (schema check) is treated the same way -- its
`.errors()` become `VerifierError`s via `verifier.errors_from_validation`.
"""
from __future__ import annotations

from pydantic import ValidationError

from axiom import config, llm_providers
from axiom.agents import verifier
from axiom.agents.contracts import ExtractionResult, VerifierResult

_SYSTEM = (
    "You extract research entities from a paper's title and abstract. Entity "
    f"kinds: {', '.join(config.ENTITY_KINDS)}. For every entity, `evidence_span` "
    "MUST be copied VERBATIM (exact substring, same characters, same case) from "
    "the abstract text given -- never paraphrase it. `confidence` is your "
    "belief the entity is correctly identified, 0.0-1.0. Only extract entities "
    "that are explicitly present in the text; never invent one."
)


def _prompt(paper_id: str, title: str | None, abstract: str | None) -> str:
    return (
        f"paper_id: {paper_id}\n"
        f"title: {title or ''}\n"
        f"abstract: {abstract or ''}\n\n"
        'Return JSON: {"paper_id": "...", "entities": [{"name": "...", '
        '"kind": "method"|"task"|"dataset"|"metric", "evidence_span": "...", '
        '"confidence": 0.0}]}. `entities` may be an empty list if none are '
        "clearly present."
    )


async def extract(
    paper_id: str,
    title: str | None,
    abstract: str | None,
    *,
    run_id: str,
    max_retries: int | None = None,
) -> ExtractionResult:
    """Extract + verify, retrying the SAME agent up to `max_retries` times.

    `max_retries` defaults to `config.AGENT_MAX_RETRIES` -- read at CALL time
    (not bound into the signature at import time) so a runtime change to the
    config value takes effect immediately.

    Raises `verifier.VerificationFailed` if no attempt passes the evidence-span
    check. A provider-layer failure that leaves no parseable result on the
    final attempt re-raises the underlying exception instead (nothing to
    report as "rejected" -- the call itself never produced output)."""
    max_retries = max_retries if max_retries is not None else config.AGENT_MAX_RETRIES
    base_prompt = _prompt(paper_id, title, abstract)
    last_errors: list = []
    last_provider_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        prompt = base_prompt
        if last_errors:
            prompt = f"{base_prompt}\n\n{verifier.format_errors_for_retry(last_errors)}"

        try:
            result = await llm_providers.complete_json(
                "fast", _SYSTEM, prompt, ExtractionResult,
                run_id=run_id, agent="extraction", agent_type="extraction",
                paper_id=paper_id, attempt=attempt,
            )
        except ValidationError as exc:
            last_errors = verifier.errors_from_validation(exc)
            last_provider_exc = exc
            verifier.log_attempt(
                run_id, "extraction",
                VerifierResult(ok=False, attempt=attempt, errors=last_errors),
                model=config.MODEL_FAST, paper_id=paper_id,
            )
            continue
        except llm_providers.LLMError as exc:
            if attempt >= max_retries:
                raise
            last_provider_exc = exc
            last_errors = []
            continue

        verdict = verifier.verify_extraction(result, abstract=abstract, attempt=attempt)
        verifier.log_attempt(run_id, "extraction", verdict, model=config.MODEL_FAST, paper_id=paper_id)
        if verdict.ok:
            return result
        last_errors = verdict.errors
        last_provider_exc = None

    if last_provider_exc is not None and not last_errors:
        raise last_provider_exc
    raise verifier.VerificationFailed("extraction", max_retries, last_errors)
