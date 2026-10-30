"""Deterministic verifier (R3), run after extraction and after enrichment.

Four checks total, split across the two call sites because each contract
only carries the fields relevant to it:

1. Pydantic-valid -- already guaranteed by the time a `ExtractionResult`/
   `EnrichmentResult` instance exists (the provider layer raises
   `pydantic.ValidationError` otherwise). `errors_from_validation()` turns
   that exception into the same `VerifierError` shape so callers have one
   uniform error list regardless of which check failed.
2. `evidence_span` is a substring of the paper's abstract -- `verify_extraction`.
3. `canonical` is in the ontology -- `verify_enrichment` (via `ontology_source`).
4. `cosine >= config.VERIFIER_COSINE_MIN` -- `verify_enrichment`.

On failure the caller (an `axiom/agents/*` worker) appends the error list to
its next prompt and retries the same agent, up to `config.AGENT_MAX_RETRIES`
attempts (R4). Every attempt -- pass or fail -- gets one `axiom/tracing.py`
line via `log_attempt`.
"""
from __future__ import annotations

from pydantic import ValidationError

from axiom import config, tracing
from axiom.agents.contracts import EnrichmentResult, ExtractionResult, VerifierError, VerifierResult


def errors_from_validation(exc: ValidationError) -> list[VerifierError]:
    """Pydantic `ValidationError.errors()` -> `VerifierError[]` (check #1)."""
    out = []
    for e in exc.errors():
        loc = ".".join(str(p) for p in e.get("loc", ())) or "_root"
        out.append(VerifierError(field=loc, code="schema", message=e.get("msg", str(e))))
    return out


def verify_extraction(result: ExtractionResult, *, abstract: str | None, attempt: int) -> VerifierResult:
    """Check #2: every `evidence_span` must literally appear in the abstract."""
    abstract_text = abstract or ""
    errors: list[VerifierError] = []
    for i, entity in enumerate(result.entities):
        if not entity.evidence_span or entity.evidence_span not in abstract_text:
            errors.append(VerifierError(
                field=f"entities[{i}].evidence_span",
                code="evidence_span",
                message=f"{entity.evidence_span!r} is not a substring of the paper abstract",
            ))
    return VerifierResult(ok=not errors, attempt=attempt, errors=errors)


def verify_enrichment(result: EnrichmentResult, *, attempt: int) -> VerifierResult:
    """Checks #3 and #4: ontology membership and the cosine floor."""
    errors: list[VerifierError] = []
    for i, entity in enumerate(result.enriched):
        if entity.ontology_source == "unmatched":
            errors.append(VerifierError(
                field=f"enriched[{i}].canonical",
                code="ontology_membership",
                message=f"{entity.canonical!r} has no ontology match (ontology_source=unmatched)",
            ))
            continue  # cosine is meaningless without a real match
        if entity.cosine < config.VERIFIER_COSINE_MIN:
            errors.append(VerifierError(
                field=f"enriched[{i}].cosine",
                code="cosine_threshold",
                message=(
                    f"cosine {entity.cosine:.4f} < VERIFIER_COSINE_MIN "
                    f"{config.VERIFIER_COSINE_MIN}"
                ),
            ))
    return VerifierResult(ok=not errors, attempt=attempt, errors=errors)


def format_errors_for_retry(errors: list[VerifierError]) -> str:
    """Render a `VerifierError[]` as a prompt nudge for the next attempt (R4)."""
    lines = [f"- [{e.code}] {e.field}: {e.message}" for e in errors]
    return (
        "Your previous attempt failed verification:\n"
        + "\n".join(lines)
        + "\nFix every issue above and return a corrected JSON response."
    )


class VerificationFailed(RuntimeError):
    """Raised by an agent after exhausting `config.AGENT_MAX_RETRIES` without
    a verified result. Carries the last error list so `axiom/agents/pipeline.py`
    can record it on the `agent_run_items` row."""

    def __init__(self, agent: str, attempts: int, errors: list[VerifierError]):
        reasons = "; ".join(f"[{e.code}] {e.field}: {e.message}" for e in errors)
        super().__init__(f"{agent}: failed verification after {attempts} attempt(s): {reasons}")
        self.agent = agent
        self.attempts = attempts
        self.errors = errors


def log_attempt(
    run_id: str, agent: str, result: VerifierResult, *,
    model: str | None = None, paper_id: str | None = None,
) -> None:
    """One trace line per verification attempt (pass or fail)."""
    tracing.log_call(
        run_id,
        agent=f"{agent}.verifier",
        model=model,
        paper_id=paper_id,
        verdict="accepted" if result.ok else "rejected",
        attempt=result.attempt,
        reasons=[f"[{e.code}] {e.field}: {e.message}" for e in result.errors],
    )
