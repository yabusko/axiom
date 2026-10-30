"""Coordinator agent (R1): one fast-tier call per paper, decides whether the
rest of the pipeline is worth running.

No verifier/self-correction loop here -- `docs/agentic/ARCHITECTURE.md`'s
data-flow diagram only wraps extraction and enrichment in the attempt<=3
verifier box; `RouteDecision` has no ontology/evidence fields to check
beyond the schema itself, which the provider layer already enforces (raises
`pydantic.ValidationError` otherwise). A paper with neither title nor
abstract is skipped deterministically, with no LLM call at all -- there is
nothing for a model to route on.
"""
from __future__ import annotations

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


def _prompt(paper_id: str, title: str | None, abstract: str | None) -> str:
    return (
        f"paper_id: {paper_id}\n"
        f"title: {title or '(missing)'}\n"
        f"abstract: {abstract or '(missing)'}\n\n"
        'Return JSON: {"paper_id": "...", "action": "full" | "title_only" | '
        '"skip", "reason": "..."}.'
    )


async def route(
    paper_id: str,
    title: str | None,
    abstract: str | None,
    *,
    run_id: str,
) -> RouteDecision:
    """Decide the route for one paper. Raises `llm_providers.LLMError` or
    `pydantic.ValidationError` on an unrecoverable provider failure --
    `axiom/agents/pipeline.py` catches that per-paper and records it on the
    run's `agent_run_items` row rather than failing the whole run."""
    if not (title or "").strip() and not (abstract or "").strip():
        return RouteDecision(paper_id=paper_id, action="skip", reason="no title or abstract")

    prompt = _prompt(paper_id, title, abstract)
    return await llm_providers.complete_json(
        "fast", _SYSTEM, prompt, RouteDecision,
        run_id=run_id, agent="coordinator",
    )
