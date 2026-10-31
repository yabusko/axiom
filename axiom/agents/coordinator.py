"""Coordinator agent (R1): decides, per paper, how much of the rest of the
pipeline is worth running.

A5: routing is rule-based FIRST, with no LLM call at all for the common
case. Three deterministic rules, checked in order:
  1. no title AND no abstract -> "skip" (nothing to extract from).
  2. abstract missing, empty, or under `config.ROUTE_TITLE_ONLY_MIN_WORDS`
     words -> "title_only" (too thin to ground a full extraction pass on).
  3. otherwise (a title exists and the abstract clears the word-count floor)
     -> "full".

These three rules are exhaustive over every (title, abstract) pair -- there
is no fourth case left for an LLM to adjudicate, so `route()` never makes a
model call today. The original one-call-per-paper LLM coordinator this
replaced is kept below as `route_via_llm()`, unused by `route()` but left in
place for a genuinely ambiguous case a future rule can't decide
deterministically (e.g. an abstract that clears the word count but reads as
template filler) -- `docs/agentic/ARCHITECTURE.md`'s data-flow diagram still
shows a coordinator call; this is what it would call. `RouteDecision.
route_source` records which path actually produced a given decision.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from axiom import config, llm_providers
from axiom.agents.contracts import RouteDecision

_SYSTEM = (
    "You are the coordinator of a research-paper extraction pipeline. Given a "
    "paper's title and abstract, decide how much work the pipeline should do "
    "on it. Return `action`: \"full\" if the abstract has enough substance to "
    "extract methods/tasks/datasets/metrics from, \"title_only\" if there is "
    "a usable title but the abstract is missing, empty, or too thin to "
    "extract anything grounded from, or \"skip\" if the paper should not be "
    "processed at all (e.g. no usable text). `reason` is one short sentence."
)


class _LLMRouteJSON(BaseModel):
    """What the model actually returns for the (currently unreachable)
    LLM fallback -- no `route_source` field (A4-style: that's pipeline
    bookkeeping the model has no business setting; `route_via_llm` stamps it
    itself after parsing)."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["full", "title_only", "skip"]
    reason: str


def _llm_prompt(title: str | None, abstract: str | None) -> str:
    return (
        f"title: {title or '(missing)'}\n"
        f"abstract: {abstract or '(missing)'}\n\n"
        'Return JSON: {"action": "full" | "title_only" | "skip", "reason": "..."}.'
    )


def _word_count(text: str) -> int:
    return len(text.split())


async def route(
    paper_id: str,
    title: str | None,
    abstract: str | None,
    *,
    run_id: str,
) -> RouteDecision:
    """Decide the route for one paper -- synchronous and deterministic (A5).

    `paper_id`/`run_id` are accepted for a uniform call signature with the
    rest of the pipeline (and so `route_via_llm`'s signature can match if it
    is ever wired in) but are not used by the rules themselves.
    """
    del paper_id, run_id  # unused by the rule path; kept for signature parity
    title_text = (title or "").strip()
    abstract_text = (abstract or "").strip()

    if not title_text and not abstract_text:
        return RouteDecision(action="skip", reason="no title or abstract", route_source="rule")

    if not abstract_text or _word_count(abstract_text) < config.ROUTE_TITLE_ONLY_MIN_WORDS:
        return RouteDecision(
            action="title_only",
            reason=f"abstract missing or under {config.ROUTE_TITLE_ONLY_MIN_WORDS} words",
            route_source="rule",
        )

    return RouteDecision(
        action="full",
        reason="title present and abstract clears the title_only word-count floor",
        route_source="rule",
    )


async def route_via_llm(
    paper_id: str,
    title: str | None,
    abstract: str | None,
    *,
    run_id: str,
) -> RouteDecision:
    """The fast-tier LLM coordinator call `route()` replaced as the default
    path (A5). Not called by `route()` today -- the three deterministic
    rules are exhaustive -- but kept callable directly for a future case
    those rules can't confidently decide. Raises `llm_providers.LLMError` or
    `pydantic.ValidationError` on an unrecoverable provider failure."""
    prompt = _llm_prompt(title, abstract)
    parsed = await llm_providers.complete_json(
        "fast", _SYSTEM, prompt, _LLMRouteJSON,
        run_id=run_id, agent="coordinator", agent_type="route", paper_id=paper_id,
    )
    return RouteDecision(action=parsed.action, reason=parsed.reason, route_source="llm")
