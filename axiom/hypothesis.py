"""Grounded hypothesis pitch generation (PBI 5, scoped per OD16).

The original plan's Step 3 assumed a per-hypothesis pipeline: a user types a
free-text thesis idea (h_syn), the system scores its geometric novelty
(s_max) and citation-scenario verdict (Dead End / Fertile Frontier), then --
only for Fertile Frontier -- a 5-node LangGraph generates a hypothesis pitch.
OD9 replaced Steps 1-2 with a corpus-level community-pair detector that has no
per-hypothesis input, so that pipeline has nothing to plug into.

OD16 scopes Step 3 down to what's real: given one of OD9's already-ranked gap
candidates, generate a grounded pitch (title, claim, method_sketch, datasets,
supporting_paper_ids) citing real papers from BOTH sides of the gap. No
LangGraph dependency -- a single LLM call (axiom/llm.py, OD14) plus a
rule-based (not LLM self-report) Verifier that checks `supporting_paper_ids`
actually belong to the two communities and there are >=2 of them, retrying on
failure. Trend context (OD10 velocity) is folded into the prompt directly
rather than as a separate agent node.

D4 (stage-2 fixes): `axiom/agents/synthesis.py::generate_hypothesis` is the one
real implementation now -- strong-tier via the provider layer, used by both
`api/main.py`'s `POST /graph/gaps/{i}/hypothesize` and `app/streamlit_app.py`'s
hypothesize button. This module's own `generate_hypothesis` (the original,
direct, Ollama-only `llm.chat_json` call) was deleted once both call sites
were repointed -- `SYSTEM`/`build_prompt`/`side_papers`/`verify`/`explain`
below are what's left: the shared prompt/verify/explain building blocks
`synthesis.py` imports and reuses rather than duplicating.
"""
from __future__ import annotations

import networkx as nx
from dataclasses import dataclass, field

from axiom.gaps import GapCandidate

MAX_RETRIES = 3
TEMPERATURE = 0.3          # per backlog: temperature <= 0.3 for the generator
MAX_PAPERS_PER_SIDE = 6    # cap prompt size; most-cited first

SYSTEM = (
    "You are pitching a candidate thesis direction that bridges two clusters "
    "of research papers which are topically related but rarely cite each "
    "other. Ground every claim in the titles/abstracts given -- never invent "
    "datasets, numbers, or papers not listed. `supporting_paper_ids` MUST "
    "include at least one paper id from EACH side, and at least 2 total."
)


@dataclass
class HypothesisPitch:
    title: str
    claim: str
    method_sketch: str
    datasets: list[str] = field(default_factory=list)
    supporting_paper_ids: list[str] = field(default_factory=list)
    disclaimer: str = "Unverified Candidate — LLM-generated pitch, not a validated research gap."


class VerificationError(RuntimeError):
    """Raised after exhausting retries without a verifiable pitch."""


def side_papers(g: nx.DiGraph, member_ids: list[str]) -> list[dict]:
    rows = [g.nodes[pid] for pid in member_ids if pid in g.nodes]
    rows_with_id = list(zip(member_ids, rows))
    rows_with_id.sort(key=lambda pair: pair[1].get("cited_by_count", 0), reverse=True)
    return [
        {"paper_id": pid, "title": d.get("title"), "year": d.get("year")}
        for pid, d in rows_with_id[:MAX_PAPERS_PER_SIDE]
    ]


def build_prompt(gap: GapCandidate, side_a: list[dict], side_b: list[dict],
                  trend_context: list[str] | None) -> str:
    def fmt_side(label: str, papers: list[dict]) -> str:
        lines = [f"  - [{p['paper_id']}] {p['title']} ({p['year']})" for p in papers]
        return f"{label}:\n" + "\n".join(lines)

    trend_line = ""
    if trend_context:
        trend_line = (f"\nRising concepts in this corpus right now: "
                       f"{', '.join(trend_context)}. Prefer connecting to these if relevant.\n")

    return (
        f"Side A — {gap.a.label}:\n<paper>\n{fmt_side('Papers', side_a)}\n</paper>\n\n"
        f"Side B — {gap.b.label}:\n<paper>\n{fmt_side('Papers', side_b)}\n</paper>\n"
        f"{trend_line}\n"
        'Return JSON: {"title": "...", "claim": "...", "method_sketch": "...", '
        '"datasets": ["..."], "supporting_paper_ids": ["..."]}. '
        "`claim` is one sentence stating the thesis direction. `method_sketch` "
        "is 2-3 sentences on how one would approach it. `datasets` names real "
        "datasets/benchmarks mentioned in the abstracts above, or an empty "
        "list if none are named — never invent one."
    )


def verify(pitch_data: dict, side_a_ids: set[str], side_b_ids: set[str]) -> HypothesisPitch | None:
    """Rule-based check (not an LLM self-report): >=2 supporting ids, both sides represented."""
    ids = [i for i in pitch_data.get("supporting_paper_ids", []) if isinstance(i, str)]
    valid_ids = [i for i in ids if i in side_a_ids or i in side_b_ids]
    has_a = any(i in side_a_ids for i in valid_ids)
    has_b = any(i in side_b_ids for i in valid_ids)
    if len(valid_ids) < 2 or not (has_a and has_b):
        return None
    title = pitch_data.get("title")
    claim = pitch_data.get("claim")
    method_sketch = pitch_data.get("method_sketch")
    if not title or not claim or not method_sketch:
        return None
    datasets = [d for d in pitch_data.get("datasets", []) if isinstance(d, str)]
    return HypothesisPitch(
        title=title, claim=claim, method_sketch=method_sketch,
        datasets=datasets, supporting_paper_ids=valid_ids,
    )


def explain(pitch_data: dict, side_a_ids: set[str], side_b_ids: set[str]) -> str:
    """D4: the structured reasons `verify()` returned `None`, for a retry
    nudge -- the same self-correction pattern `axiom/agents/verifier.py`
    uses (send the model its ACTUAL failure, not a generic reminder)."""
    ids = [i for i in pitch_data.get("supporting_paper_ids", []) if isinstance(i, str)]
    valid_ids = [i for i in ids if i in side_a_ids or i in side_b_ids]
    has_a = any(i in side_a_ids for i in valid_ids)
    has_b = any(i in side_b_ids for i in valid_ids)
    reasons = []
    if len(valid_ids) < 2:
        reasons.append(
            f"supporting_paper_ids needs >=2 valid ids from the lists above "
            f"(got {len(valid_ids)}: {valid_ids})"
        )
    if not has_a:
        reasons.append("no valid id from Side A is included")
    if not has_b:
        reasons.append("no valid id from Side B is included")
    for field in ("title", "claim", "method_sketch"):
        if not pitch_data.get(field):
            reasons.append(f"`{field}` is missing or empty")
    if not reasons:
        reasons.append("unknown verification failure")
    return "; ".join(reasons)
