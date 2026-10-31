"""Paper summarization (OD14). Produces a 3-bullet summary of a paper's
abstract, grounded only in the given text -- used by the Reading List
(Task 7.4) and, later, the hypothesis pipeline's Ingest_Summarizer node
(PBI 5).

Routed through `axiom/llm_providers.py` (the one pooled LLM client, picking
Anthropic if `ANTHROPIC_API_KEY` is set else local Ollama -- same
provider-selection and spend/trace ledger as the rest of the agentic
pipeline) rather than calling `axiom/llm.py` directly, which this module did
until stage-3 cleanup item A.1.6.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from axiom import llm_providers

_SYSTEM = (
    "You summarize a single research paper abstract into exactly 3 bullet "
    "points. Every bullet must be a claim directly supported by the abstract "
    "text given -- never add outside knowledge, invented numbers, or claims "
    "not present in the text. Keep each bullet under 25 words."
)


class _SummaryBullets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bullets: list[str]


@dataclass
class PaperSummary:
    paper_id: str
    bullets: list[str]


def _prompt(title: str, abstract: str) -> str:
    return (
        f"Title: {title}\n\nAbstract: {abstract}\n\n"
        'Return JSON: {"bullets": ["...", "...", "..."]} -- exactly 3 bullets.'
    )


async def summarize_paper_async(paper_id: str, title: str | None, abstract: str | None) -> PaperSummary:
    if not abstract:
        raise ValueError(f"paper {paper_id} has no abstract to summarize")
    result = await llm_providers.complete_json(
        "fast", _SYSTEM, _prompt(title or "", abstract), _SummaryBullets,
        agent="summarize", agent_type="summarize", paper_id=paper_id,
    )
    bullets = [b for b in result.bullets if b][:3]
    if not bullets:
        raise llm_providers.LLMParseError(f"model returned no usable bullets for {paper_id}")
    return PaperSummary(paper_id=paper_id, bullets=bullets)


def summarize_paper(paper_id: str, title: str | None, abstract: str | None) -> PaperSummary:
    """Sync wrapper -- every caller today (Streamlit, the API endpoint) is sync."""
    return asyncio.run(summarize_paper_async(paper_id, title, abstract))
