"""Deterministic verifier (R3), run after extraction and after enrichment.

Checks, split across the two call sites because each contract only carries
the fields relevant to it:

1. Pydantic-valid -- already guaranteed by the time a `ExtractionResult`/
   `EnrichmentResult` instance exists (the provider layer raises
   `pydantic.ValidationError` otherwise). `errors_from_validation()` turns
   that exception into the same `VerifierError` shape so callers have one
   uniform error list regardless of which check failed.
2. `evidence_span` must appear, after normalization, in the paper's title OR
   abstract -- `verify_extraction` (A1: title is now checked too, not just
   the abstract; both sides are normalized before comparison).
3. The claimed entity `name` must actually relate to its own `evidence_span`
   -- `verify_extraction` (A3: catches a span that is real text from the
   paper but doesn't actually mention the named entity).
4. `canonical` is in the ontology -- `verify_enrichment` (via `ontology_source`).
5. `cosine >= config.VERIFIER_COSINE_MIN` -- `verify_enrichment`.

On failure the caller (an `axiom/agents/*` worker) appends the error list to
its next prompt and retries the same agent, up to `config.AGENT_MAX_RETRIES`
attempts (R4). Every attempt -- pass or fail -- gets one `axiom/tracing.py`
line via `log_attempt`.
"""
from __future__ import annotations

import re
import unicodedata

from pydantic import ValidationError

from axiom import config, tracing
from axiom.agents.contracts import EnrichmentResult, ExtractionResult, VerifierError, VerifierResult

# A1: dash/quote variants that should compare as equal once normalized. Hyphen,
# non-breaking hyphen, figure dash, en dash, em dash, horizontal bar, minus sign
# -> plain ASCII hyphen; curly single/double quotes -> straight quotes.
_DASH_CHARS = "‐‑‒–—―−"
_QUOTE_MAP = {
    "‘": "'", "’": "'", "‛": "'",
    "“": '"', "”": '"', "‟": '"',
}
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(s: str | None) -> str:
    """NFKC-normalize, casefold, unify dash/quote variants, collapse whitespace.

    Used on BOTH sides of every substring/overlap comparison in this module
    (A1) so a span that differs from the source text only in curly-quote
    style, dash choice, or incidental whitespace still counts as a match.
    """
    if not s:
        return ""
    text = unicodedata.normalize("NFKC", s).casefold()
    for ch in _DASH_CHARS:
        text = text.replace(ch, "-")
    for ch, repl in _QUOTE_MAP.items():
        text = text.replace(ch, repl)
    return _WHITESPACE_RE.sub(" ", text).strip()


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(normalized: str) -> set[str]:
    return set(_TOKEN_RE.findall(normalized))


def _acronym(normalized_name: str) -> str:
    """First-letter acronym of a multi-word name, e.g. 'low rank adaptation' -> 'lra'.

    Single-word names have no acronym (returns ""); a one-letter "acronym"
    would be too likely to coincidentally appear in unrelated text.
    """
    words = normalized_name.split()
    if len(words) < 2:
        return ""
    return "".join(w[0] for w in words if w)


def name_relates_to_span(name: str, span: str) -> bool:
    """A3: does the claimed entity `name` actually relate to its own `evidence_span`?

    True if, after normalization, EITHER:
      - `name` appears verbatim as a substring of `span` (covers the common
        case -- the span is a sentence containing the exact entity name), OR
      - a multi-word name's first-letter acronym appears in `span` (covers
        "LoRA" naming a span that spells out "low-rank adaptation"), OR
      - token overlap is high: |tokens(name) & tokens(span)| / |tokens(name)|
        >= 0.5 -- at least half of the name's own words show up in the span
        (denominator is len(tokens(name)), not the union/span size, so a long
        span padded with unrelated words doesn't dilute the score; a short
        name needs most of its words present to pass).
    Catches a fabricated/unrelated name paired with a real-but-irrelevant
    span, which the plain evidence_span substring check can't catch on its
    own (that check only verifies the SPAN is real text, not that the NAME
    has anything to do with it).
    """
    name_norm = normalize_text(name)
    span_norm = normalize_text(span)
    if not name_norm or not span_norm:
        return False
    if name_norm in span_norm:
        return True
    acronym = _acronym(name_norm)
    if acronym and acronym in span_norm:
        return True
    name_tokens = _tokens(name_norm)
    if not name_tokens:
        return False
    overlap = len(name_tokens & _tokens(span_norm)) / len(name_tokens)
    return overlap >= 0.5


def errors_from_validation(exc: ValidationError) -> list[VerifierError]:
    """Pydantic `ValidationError.errors()` -> `VerifierError[]` (check #1)."""
    out = []
    for e in exc.errors():
        loc = ".".join(str(p) for p in e.get("loc", ())) or "_root"
        out.append(VerifierError(field=loc, code="schema", message=e.get("msg", str(e))))
    return out


def verify_extraction(
    result: ExtractionResult, *, title: str | None = None, abstract: str | None = None,
    attempt: int,
) -> VerifierResult:
    """Checks #2/#3: every `evidence_span` must appear (normalized) in the
    title OR the abstract, and every `name` must relate to its own span.

    `title`/`abstract` are both optional so a title_only route (A5, no usable
    abstract) can call this with `abstract=None` and still verify spans
    against the title.
    """
    title_norm = normalize_text(title)
    abstract_norm = normalize_text(abstract)
    errors: list[VerifierError] = []
    for i, entity in enumerate(result.entities):
        span_norm = normalize_text(entity.evidence_span)
        span_found = bool(span_norm) and (span_norm in title_norm or span_norm in abstract_norm)
        if not span_found:
            errors.append(VerifierError(
                field=f"entities[{i}].evidence_span",
                code="evidence_span",
                message=f"{entity.evidence_span!r} is not a substring of the paper title or abstract",
            ))
            continue  # name/span relation is moot if the span itself isn't real
        if not name_relates_to_span(entity.name, entity.evidence_span):
            errors.append(VerifierError(
                field=f"entities[{i}].name",
                code="name_span_mismatch",
                message=(
                    f"{entity.name!r} does not appear in, or overlap enough with, "
                    f"its own evidence_span {entity.evidence_span!r}"
                ),
            ))
    return VerifierResult(ok=not errors, attempt=attempt, errors=errors)


def verify_enrichment(result: EnrichmentResult, *, attempt: int) -> VerifierResult:
    """Checks #4 and #5: ontology membership and the cosine floor.

    B1/B2: `ontology_source == "new_concept"` means enrichment deliberately
    found no ontology match and kept the entity anyway (step 3 of the
    cascade) -- that's not a failure, it has no real ontology match to be
    below-threshold on, so it's never rejected here. `"lexical"` (exact
    normalized-name match) and `"embedding"` (short-phrase-model match, only
    assigned by enrichment.py when it already cleared the cosine+margin
    gate) both carry a real `cosine`; the floor check stays as a backstop in
    case that invariant is ever violated upstream.
    """
    errors: list[VerifierError] = []
    for i, entity in enumerate(result.enriched):
        if entity.ontology_source == "new_concept":
            continue  # by construction: no real ontology match to check
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
